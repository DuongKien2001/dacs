import argparse
import os
import sys
import random
import timeit
import datetime
import copy 
import numpy as np
import pickle
import scipy.misc
from core.configs import cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from core.utils.prototype_dist_estimator import prototype_dist_estimator

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from core.utils.loss import PrototypeContrastiveLoss

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate

import time

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    return parser.parse_args()


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('dacs/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('dacs/', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def prototype_dist_init(cfg, trainloader, model):
    feature_num = 2048
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    torch.cuda.empty_cache()

    iteration = 0
    model.eval()
    end = time.time()
    start_time = time.time()
    max_iters = len(trainloader)
    
    with torch.no_grad():
        for i, (src_input, src_label, _, _) in enumerate(trainloader):
            data_time = time.time() - end

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            src_out, src_feat = model(src_input)
            B, N, Hs, Ws = src_feat.size()
            _, C, _, _ = src_out.size()

            # source mask: downsample the ground-truth label
            src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )

            # feature level
            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            feat_estimator.update(features=src_feat.detach().clone(), labels=src_mask)

            # output level
            src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, C)
            out_estimator.update(features=src_out.detach().clone(), labels=src_mask)

            batch_time = time.time() - end
            end = time.time()
            #meters.update(time=batch_time, data=data_time)

            iteration = iteration + 1
            #eta_seconds = meters.time.global_avg * (max_iters - iteration)
            #eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


            if iteration == max_iters:
                feat_estimator.save(name='prototype_feat_dist.pth')
                out_estimator.save(name='prototype_out_dist.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_time / max_iters))

def main():
    print(config)
    list_name = []
    best_mIoU = 0
    feature_num = 2048

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)
    
    # load pretrained parameters
    #saved_state_dict = torch.load(args.restore_from)
        # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
    
    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    
    
    cudnn.benchmark = True
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    if cfg.SOLVER.MULTI_LEVEL:
        out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)
    pcl_criterion_src = PrototypeContrastiveLoss(cfg)
    pcl_criterion_tgt = PrototypeContrastiveLoss(cfg)
    
    if dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if labeled_samples is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = labeled_samples
        print('Training on number of samples:', partial_size)
        np.random.seed(random_seed)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    #New loader for Domain transfer
    a = ['21821.png', '12100.png', '16518.png', '13121.png', '17825.png', '08290.png', '03581.png', '12398.png', '17961.png', '17521.png', '12997.png', '06911.png', '17388.png', '05069.png', '15891.png', '23251.png', '08317.png', '02451.png', '01531.png', '23723.png', '21027.png', '19210.png', '04638.png', '22838.png', '09666.png', '15597.png', '03164.png', '03814.png', '12712.png', '14447.png', '13338.png', '12465.png', '16283.png', '24876.png', '19674.png', '08898.png', '18370.png', '07857.png', '12355.png', '14993.png', '24575.png', '09647.png', '01660.png', '02354.png', '20827.png', '12582.png', '09893.png', '11645.png', '08180.png', '21054.png', '04849.png', '12105.png', '05442.png', '17174.png', '15492.png', '14218.png', '13715.png', '17300.png', '15771.png', '09164.png', '08738.png', '00106.png', '20220.png', '03659.png', '03875.png', '04610.png', '07419.png', '18254.png', '21898.png', '16875.png', '04054.png', '24090.png', '22808.png', '08348.png', '17142.png', '12988.png', '20257.png', '17982.png', '01024.png', '18136.png', '08968.png', '10264.png', '15204.png', '08952.png', '11303.png', '00170.png', '08046.png', '12304.png', '13457.png', '03400.png', '04642.png', '18942.png', '22428.png', '09801.png', '09386.png', '16478.png', '09623.png', '17988.png', '03594.png', '16033.png', '21246.png', '10244.png', '05281.png', '24567.png', '02081.png', '24564.png', '10409.png', '00936.png', '00563.png', '22132.png', '20069.png', '10882.png', '00858.png', '07126.png', '04982.png', '06040.png', '07897.png', '17367.png', '02572.png', '17165.png', '20554.png', '20146.png', '00972.png', '03902.png', '09355.png', '01489.png', '22813.png', '14145.png', '15382.png', '04427.png', '13234.png', '13576.png', '20038.png', '24631.png', '07394.png', '01940.png', '05182.png', '16930.png', '02612.png', '17109.png', '04284.png', '10214.png', '23610.png', '21143.png', '02509.png', '12441.png', '20529.png', '15601.png', '12141.png', '11885.png', '02730.png', '10938.png', '07396.png', '05123.png', '12816.png', '09872.png', '11763.png', '07783.png', '24193.png', '23495.png', '13485.png', '00045.png', '03871.png', '12069.png', '08551.png', '22845.png', '15467.png', '13984.png', '21002.png', '11367.png', '01209.png', '10943.png', '00386.png', '20389.png', '18760.png', '12097.png', '08399.png', '02993.png', '10622.png', '15110.png', '08271.png', '08896.png', '24053.png', '19654.png', '10907.png', '17301.png', '10319.png', '03352.png', '07933.png', '21083.png', '14869.png', '05028.png', '03184.png', '05755.png', '19212.png', '08319.png', '03310.png', '05470.png', '24527.png', '18528.png', '00258.png', '20402.png', '19479.png', '02256.png', '05035.png', '15411.png', '01757.png', '15274.png', '03691.png', '06691.png', '11726.png', '07376.png', '15795.png', '21252.png', '00710.png', '04269.png', '09221.png', '13724.png', '08681.png', '09787.png', '05383.png', '14583.png', '12274.png', '18719.png', '13141.png', '17076.png', '18655.png', '12385.png', '03732.png', '01801.png', '07919.png', '14911.png', '04882.png', '01578.png', '00060.png', '03049.png', '19177.png', '22496.png', '13846.png', '11884.png', '04877.png', '12468.png', '04756.png', '07166.png', '24300.png', '07907.png', '12065.png', '09649.png', '10967.png', '24345.png', '23630.png', '15258.png', '11079.png', '01840.png', '21676.png', '22150.png', '16634.png', '19118.png', '05088.png', '06100.png', '09597.png', '07530.png', '17895.png', '13703.png', '03644.png', '17956.png', '14298.png', '00151.png', '21948.png', '19783.png', '16348.png', '20514.png', '02521.png', '16655.png', '16779.png', '08847.png', '24939.png', '11428.png', '09412.png', '19902.png', '00422.png', '23455.png', '03621.png', '22555.png', '08025.png', '21597.png', '01150.png', '18531.png', '15016.png', '13696.png', '10803.png', '24480.png', '05006.png', '12540.png', '21799.png', '11526.png', '15623.png', '24731.png', '13831.png', '18280.png', '11509.png', '14939.png', '02930.png', '23635.png', '22781.png', '23874.png', '10696.png', '19844.png', '12843.png', '01767.png', '02337.png', '01212.png', '01052.png', '18339.png', '11378.png', '14556.png', '08725.png', '09830.png', '03907.png', '01764.png', '10640.png', '00300.png', '10314.png', '06510.png', '01268.png', '13300.png', '09591.png', '15577.png', '10107.png', '10541.png', '12523.png', '09887.png', '08661.png', '21929.png', '24670.png', '22610.png', '12579.png', '14198.png', '22899.png', '11836.png', '14898.png', '21603.png', '02375.png', '16773.png', '14765.png', '20646.png', '11128.png', '20822.png', '05199.png', '06977.png', '05928.png', '17540.png', '18539.png', '05321.png', '05388.png', '21329.png', '08003.png', '20892.png', '19576.png', '24911.png', '20771.png', '14456.png', '19096.png', '01718.png', '23642.png', '04263.png', '04509.png', '10436.png', '22682.png', '20007.png', '22028.png', '01201.png', '12640.png', '13254.png', '16668.png', '11625.png', '17904.png', '08200.png', '05000.png', '22497.png', '18907.png', '16716.png', '06802.png', '08954.png', '03360.png', '06010.png', '07891.png', '15488.png', '05784.png', '12737.png', '20681.png', '23712.png', '12287.png', '12984.png', '05130.png', '03489.png', '02886.png', '17890.png', '03056.png', '10620.png', '16842.png', '00361.png', '03771.png', '23108.png', '23706.png', '17520.png', '14306.png', '05692.png', '09359.png', '09233.png', '24278.png', '04367.png', '10572.png', '09189.png', '05258.png', '11852.png', '06843.png', '10651.png', '13704.png', '20307.png', '06425.png', '10413.png', '12312.png', '00826.png', '19161.png', '19362.png', '01358.png', '08636.png', '13354.png', '11803.png', '17953.png', '15715.png', '00282.png', '14459.png', '15662.png', '04856.png', '11004.png', '20367.png', '22378.png', '01974.png', '11394.png', '12950.png', '23817.png', '19502.png', '00328.png', '12951.png', '08345.png', '21839.png', '18239.png', '17406.png', '06661.png', '12071.png', '13130.png', '02879.png', '20902.png', '04250.png', '15484.png', '12718.png', '10027.png', '15710.png', '12432.png', '23179.png', '03335.png', '23454.png', '16234.png', '20193.png', '06387.png', '13165.png', '17846.png', '06127.png', '21406.png', '19049.png', '10349.png', '19600.png', '24199.png', '11938.png', '04973.png', '03135.png', '01047.png', '07447.png', '17101.png', '12123.png', '00884.png', '09788.png', '22304.png', '18563.png', '05553.png', '11965.png', '11533.png', '16040.png', '19948.png', '17295.png', '21509.png', '07464.png', '07677.png', '15129.png', '12685.png', '06780.png', '22991.png', '05013.png', '01411.png', '14333.png', '09184.png', '18702.png', '04323.png', '16361.png', '11197.png', '07885.png', '02013.png', '07136.png', '17129.png', '00154.png', '10627.png', '24089.png', '07507.png', '03841.png', '18456.png', '01899.png', '14168.png', '19203.png', '19903.png', '08228.png', '22895.png', '00778.png', '06187.png', '09829.png', '07538.png', '22782.png', '03906.png', '01983.png', '10062.png', '15112.png', '06641.png', '07021.png', '20865.png', '20779.png', '24191.png', '23931.png', '00774.png', '11127.png', '08859.png', '20698.png', '04752.png', '13934.png', '23028.png', '07968.png', '01802.png', '22277.png', '08338.png', '22937.png', '16980.png', '18425.png', '11136.png', '04230.png', '10216.png', '06440.png', '01325.png', '17160.png', '13253.png', '01016.png', '05593.png', '03739.png', '07828.png', '05984.png', '22755.png', '22495.png', '07639.png', '18742.png', '19348.png', '07729.png', '16992.png', '03813.png', '16370.png', '01836.png', '02242.png', '04276.png', '13320.png', '00388.png', '16850.png', '22305.png', '04211.png', '02033.png', '03579.png', '00329.png', '12201.png', '16360.png', '19402.png', '11735.png', '15521.png', '02636.png', '24694.png', '11440.png', '18459.png', '23491.png', '04361.png', '22257.png', '05101.png', '01420.png', '12514.png', '19258.png', '15193.png', '14561.png', '22666.png', '06170.png', '22331.png', '07941.png', '22897.png', '10910.png', '04301.png', '22355.png', '04499.png', '21567.png', '07870.png', '09018.png', '17124.png', '17936.png', '17238.png', '22225.png', '15565.png', '23638.png', '18264.png', '03037.png', '20417.png', '04267.png', '13924.png', '06057.png', '17564.png', '18437.png', '12000.png', '06639.png', '17780.png', '14116.png', '14607.png', '16446.png', '15433.png', '05074.png', '00129.png', '15592.png', '09499.png', '02496.png', '24499.png', '07007.png', '20832.png', '05843.png', '00284.png', '16016.png', '11549.png', '09128.png', '11899.png', '16215.png', '12070.png', '15527.png', '12871.png', '06784.png', '03478.png', '17065.png', '01751.png', '08263.png', '07779.png', '19382.png', '09126.png', '23162.png', '18922.png', '10531.png', '12884.png', '22251.png', '19689.png', '14121.png', '03507.png', '16680.png', '16826.png', '19564.png', '05914.png', '00896.png', '09136.png', '03880.png', '07263.png', '10793.png', '15401.png', '13902.png', '19193.png', '03687.png', '14451.png', '00556.png', '18782.png', '20856.png', '22410.png', '24522.png', '21500.png', '08554.png', '11236.png', '24083.png', '03931.png', '21673.png', '08031.png', '03171.png', '19544.png', '24211.png', '12911.png', '24771.png', '05673.png', '11685.png', '08856.png', '23210.png', '03232.png', '10846.png', '01400.png', '20432.png', '04677.png', '10003.png', '14354.png', '22462.png', '08451.png', '09727.png', '06169.png', '13991.png', '14630.png', '10675.png', '09674.png', '03263.png', '21660.png', '13003.png', '14581.png', '19578.png', '05245.png', '15197.png', '03486.png', '21138.png', '06496.png', '15448.png', '08461.png', '02490.png', '20995.png', '04438.png', '05188.png', '24087.png', '02925.png', '08553.png', '11108.png', '12629.png', '01680.png', '19992.png', '15010.png', '24450.png', '11644.png', '19746.png', '01303.png', '04419.png', '17889.png', '09920.png', '17693.png', '06433.png', '21348.png', '23520.png', '24603.png', '15512.png', '24164.png', '16335.png', '04056.png', '20945.png', '02716.png', '13445.png', '05627.png', '05062.png', '12529.png', '22084.png', '20959.png', '06826.png', '07363.png', '18779.png', '12632.png', '17677.png', '02101.png', '08654.png', '03207.png', '10498.png', '01713.png', '01046.png', '21267.png', '13459.png', '01221.png', '07352.png', '22989.png', '16559.png', '06473.png', '18833.png', '06871.png', '10789.png', '17983.png', '06979.png', '06864.png', '06329.png', '21289.png', '20049.png', '13931.png', '21141.png', '22287.png', '14266.png', '21154.png', '16884.png', '21801.png', '14253.png', '13380.png', '00190.png', '13310.png', '20493.png', '11931.png', '20531.png', '20619.png', '19414.png', '24518.png', '17455.png', '17716.png', '18544.png', '15788.png', '10091.png', '00835.png', '18332.png', '00387.png', '06102.png', '14568.png', '04776.png', '13296.png', '20637.png', '02485.png', '16928.png', '13297.png', '11586.png', '02285.png', '24853.png', '07621.png', '04189.png', '20046.png', '24070.png', '19401.png', '04559.png', '15099.png', '12433.png', '01432.png', '17384.png', '16713.png', '19709.png', '12263.png', '10453.png', '24705.png', '05184.png', '13206.png', '24316.png', '00914.png', '10749.png', '11435.png', '14836.png', '02855.png', '05320.png', '24874.png', '08327.png', '20232.png', '16088.png', '06326.png', '18851.png', '14702.png', '00953.png', '07851.png', '10564.png', '04517.png', '10516.png', '18389.png', '11372.png', '23749.png', '05170.png', '24076.png', '16271.png', '16859.png', '16990.png', '16282.png', '08790.png', '13975.png', '19960.png', '13255.png', '22309.png', '20708.png', '16606.png', '00638.png', '20362.png', '15787.png', '14037.png', '09707.png', '23596.png', '03089.png', '00679.png', '14559.png', '13110.png', '10483.png', '01607.png', '24588.png', '22810.png', '11219.png', '07566.png', '05859.png', '13986.png', '10616.png', '08704.png', '00240.png', '23307.png', '13961.png', '23898.png', '08424.png', '08043.png', '06068.png', '01157.png', '10374.png', '03328.png', '07718.png', '15173.png', '16818.png', '02245.png', '01474.png', '22718.png', '16569.png', '07097.png', '08419.png', '19766.png', '05292.png', '24253.png', '09850.png', '05618.png', '16020.png', '12138.png', '20453.png', '10088.png', '11407.png', '09312.png', '08582.png', '03915.png', '04943.png', '02525.png', '06693.png', '05750.png', '03527.png', '09513.png', '21935.png', '08736.png', '20055.png', '03666.png', '24380.png', '24517.png', '02601.png', '04953.png', '07420.png', '08214.png', '18927.png', '02851.png', '19938.png', '02989.png', '02711.png', '22106.png', '00276.png', '19622.png', '03225.png', '10383.png', '01188.png', '17484.png', '23516.png', '19507.png', '20358.png', '15092.png', '07279.png', '14691.png', '15912.png', '08773.png', '04430.png', '04939.png', '19794.png', '05440.png', '02172.png', '02179.png', '01225.png', '17469.png', '12293.png', '18073.png', '12051.png', '20538.png', '09599.png', '10039.png', '05957.png', '06360.png', '18036.png', '08192.png', '10176.png', '17942.png', '07895.png', '05806.png', '07896.png', '23388.png', '06923.png', '17980.png', '02835.png', '12055.png', '09605.png', '03820.png', '03588.png', '10609.png', '20608.png', '07271.png', '17805.png', '09663.png', '00991.png', '20238.png', '02063.png', '14045.png', '02151.png', '00680.png', '03585.png', '14510.png', '07944.png', '23976.png', '19612.png', '05276.png', '03833.png', '10726.png', '14431.png', '00764.png', '06920.png', '11729.png', '22136.png', '07562.png', '04050.png', '03767.png', '14875.png', '09767.png', '14012.png', '03878.png', '00808.png', '12296.png', '18685.png', '01424.png', '21323.png', '05444.png', '12578.png', '23803.png', '10840.png', '09895.png', '11320.png', '20889.png', '18556.png', '01919.png', '13747.png', '10290.png', '24466.png', '14694.png', '01842.png', '21292.png', '13199.png', '06851.png', '22345.png', '16996.png', '04662.png', '14345.png', '04891.png', '13632.png', '16403.png', '13548.png', '16291.png', '08183.png', '14589.png', '22065.png', '21170.png', '16712.png', '01979.png', '00675.png', '09409.png', '21858.png', '08314.png', '07991.png', '19192.png', '04708.png', '08341.png', '11527.png', '09463.png', '02330.png', '00868.png', '02484.png', '00221.png', '16498.png', '04664.png', '10569.png', '09958.png', '00583.png', '04548.png', '19166.png', '11106.png', '24002.png', '03862.png', '15100.png', '22424.png', '12257.png', '23972.png', '13689.png', '24254.png', '12154.png', '16697.png', '03923.png', '03369.png', '21184.png', '09658.png', '15424.png', '10527.png', '02309.png', '14015.png', '06250.png', '15295.png', '05769.png', '24790.png', '04523.png', '06884.png', '07609.png', '01520.png', '09205.png', '03196.png', '04371.png', '02089.png', '14971.png', '01688.png', '21505.png', '15721.png', '24459.png', '02668.png', '02298.png', '23489.png', '14283.png', '16902.png', '00115.png', '06810.png', '03470.png', '05541.png', '18754.png', '13466.png', '18116.png', '23563.png', '24483.png', '19753.png', '08333.png', '20561.png', '22590.png', '15863.png', '04842.png', '13593.png', '12709.png', '00560.png', '23899.png', '15666.png', '06855.png', '13892.png', '04853.png', '17473.png', '05250.png', '08515.png', '03775.png', '01923.png', '22671.png', '19801.png', '18263.png', '20101.png', '23033.png', '01079.png', '09351.png', '24011.png', '05698.png', '07322.png', '09215.png', '20273.png', '23400.png', '08467.png', '23558.png', '00135.png', '16085.png', '11323.png', '18212.png', '03637.png', '06380.png', '16523.png', '20217.png', '24422.png', '09979.png', '22873.png', '08565.png', '06972.png', '05946.png', '11741.png', '24263.png', '08832.png', '03831.png', '00342.png', '05222.png', '12858.png', '01112.png', '20227.png', '01222.png', '03342.png', '14922.png', '11910.png', '14914.png', '06293.png', '16119.png', '07824.png', '20426.png', '19398.png', '13562.png', '03799.png', '03021.png', '21059.png', '21823.png', '20391.png', '19852.png', '19900.png', '17126.png', '14884.png', '14913.png', '24937.png', '15012.png', '24194.png', '13874.png', '21465.png', '09182.png', '18590.png', '12730.png', '06286.png', '08824.png', '01988.png', '22727.png', '24555.png', '02431.png', '19759.png', '00146.png', '19943.png', '15998.png', '01571.png', '04207.png', '15352.png', '23951.png', '23829.png', '20323.png', '09665.png', '21958.png', '22104.png', '08018.png', '21457.png', '04418.png', '23772.png', '10787.png', '09405.png', '14544.png', '08737.png', '10146.png', '07029.png', '16397.png', '19599.png', '23019.png', '10090.png', '02539.png', '02441.png', '19082.png', '14633.png', '00200.png', '16845.png', '14994.png', '11093.png', '22152.png', '11780.png', '18944.png', '03945.png', '12082.png', '14397.png', '13691.png', '02991.png', '17809.png', '08949.png', '01670.png', '11457.png', '01918.png', '00810.png', '06140.png', '18201.png', '16592.png', '21382.png', '02819.png', '23485.png', '00948.png', '18919.png', '18290.png', '10814.png', '07341.png', '06676.png', '19700.png', '19528.png', '16332.png', '06403.png', '23194.png', '23484.png', '05701.png', '10949.png', '21274.png', '01480.png', '02102.png', '12115.png', '08344.png', '23920.png', '20314.png', '06208.png', '15180.png', '09947.png', '04368.png', '19949.png', '00798.png', '24144.png', '20992.png', '06117.png', '03958.png', '17189.png', '17581.png', '03188.png', '06483.png', '15229.png', '07489.png', '21130.png', '03129.png', '02105.png', '03394.png', '01127.png', '12539.png', '16069.png', '03744.png', '19174.png', '01952.png', '06115.png', '21316.png', '13243.png', '16760.png', '05845.png', '19070.png', '05888.png', '12896.png', '13943.png', '17692.png', '20168.png', '09323.png', '24958.png', '12678.png', '05956.png', '24207.png', '13179.png', '21830.png', '12699.png', '24147.png', '05133.png', '18558.png', '18166.png', '07628.png', '21189.png', '05242.png', '08696.png', '11996.png', '02793.png', '08450.png', '03760.png', '15178.png', '19985.png', '08383.png', '08413.png', '22066.png', '01703.png', '21997.png', '00644.png', '19982.png', '15903.png', '03634.png', '03449.png', '20507.png', '01794.png', '10103.png', '21010.png', '24057.png', '15987.png', '22181.png', '08641.png', '15135.png', '08169.png', '13401.png', '09352.png', '01318.png', '21241.png', '13959.png', '23032.png', '21330.png', '21071.png', '12237.png', '00242.png', '02811.png', '09314.png', '14984.png', '00243.png', '09030.png', '00624.png', '22657.png', '02644.png', '20770.png', '19332.png', '20396.png', '07532.png', '13675.png', '24673.png', '21824.png', '08321.png', '03240.png', '07504.png', '11066.png', '12233.png', '05033.png', '16350.png', '14461.png', '07577.png', '10101.png', '17425.png', '15900.png', '08151.png', '14211.png', '24256.png', '02414.png', '13053.png', '06711.png', '15540.png', '09306.png', '21407.png', '14610.png', '15483.png', '16628.png', '02223.png', '21109.png', '03399.png', '11755.png', '12556.png', '03454.png', '13753.png', '20821.png', '17934.png', '14025.png', '23496.png', '23205.png', '13681.png', '14132.png', '09175.png', '07020.png', '13524.png', '04611.png', '13952.png', '17445.png', '16793.png', '02749.png', '09226.png', '07734.png', '24109.png', '21298.png', '20056.png', '08240.png', '18986.png', '07529.png', '01207.png', '06423.png', '06757.png', '05151.png', '00014.png', '14466.png', '16500.png', '00371.png', '08539.png', '11787.png', '23022.png', '14503.png', '06053.png', '05836.png', '08930.png', '22643.png', '24409.png', '10373.png', '24861.png', '07139.png', '24390.png', '16429.png', '06238.png', '10282.png', '12002.png', '16539.png', '13617.png', '23569.png', '10153.png', '18325.png', '10408.png', '20073.png', '15318.png', '01388.png', '00302.png', '11338.png', '10316.png', '18098.png', '09079.png', '20312.png', '09772.png', '14527.png', '13530.png', '05815.png', '03746.png', '15116.png', '12052.png', '15371.png', '04470.png', '19965.png', '08237.png', '00719.png', '05708.png', '03980.png', '09283.png', '06621.png', '04929.png', '15921.png', '15464.png', '22688.png', '11349.png', '24295.png', '06584.png', '07987.png', '11285.png', '04131.png', '04812.png', '18277.png', '15625.png', '13133.png', '16660.png', '03150.png', '14029.png', '15288.png', '23199.png', '08600.png', '21198.png', '19755.png', '10499.png', '15134.png', '15291.png', '11297.png', '22098.png', '24762.png', '05248.png', '13385.png', '19780.png', '14803.png', '16072.png', '14730.png', '00861.png', '08714.png', '02834.png', '08313.png', '14245.png', '16431.png', '23812.png', '12574.png', '06553.png', '19988.png', '12451.png', '20274.png', '04805.png', '17358.png', '04651.png', '18103.png', '03371.png', '10143.png', '14980.png', '06299.png', '17409.png', '20245.png', '02779.png', '04965.png', '07694.png', '16561.png', '24395.png', '16468.png', '02095.png', '09267.png', '24904.png', '24103.png', '24786.png', '12639.png', '17313.png', '17875.png', '07024.png', '04839.png', '13942.png', '02916.png', '07069.png', '01698.png', '01966.png', '23279.png', '22532.png', '20823.png', '01062.png', '17345.png', '17918.png', '16971.png', '22402.png', '20812.png', '23511.png', '08689.png', '15020.png', '15028.png', '22166.png', '24432.png', '07118.png', '21015.png', '08493.png', '06455.png', '00407.png', '08865.png', '11667.png', '02563.png', '21035.png', '15332.png', '17382.png', '21123.png', '04719.png', '03580.png', '03452.png', '21070.png', '03283.png', '06682.png', '18733.png', '04769.png', '15673.png', '04785.png', '22193.png', '13210.png', '08258.png', '02753.png', '12799.png', '19093.png', '23273.png', '11989.png', '23270.png', '15280.png', '00272.png', '05485.png', '09289.png', '14600.png', '13388.png', '21793.png', '20909.png', '19034.png', '16749.png', '23613.png', '24308.png', '17501.png', '09815.png', '14566.png', '10236.png', '09376.png', '18725.png', '00988.png', '23986.png', '22872.png', '00558.png', '04064.png', '11752.png', '09466.png', '03598.png', '04003.png', '05756.png', '14482.png', '13861.png', '23846.png', '21456.png', '17579.png', '05876.png', '24899.png', '18690.png', '21047.png', '19172.png', '03163.png', '05397.png', '09278.png', '08443.png', '18060.png', '15435.png', '14178.png', '03936.png', '18505.png', '18929.png', '03725.png', '21342.png', '01238.png', '10366.png', '19768.png', '05798.png', '11149.png', '23245.png', '21204.png', '14046.png', '23680.png', '08846.png', '12195.png', '22759.png', '22603.png', '24614.png', '23198.png', '16542.png', '03900.png', '19954.png', '07415.png', '21746.png', '24953.png', '02340.png', '20694.png', '03253.png', '24895.png', '07682.png', '19458.png', '20964.png', '17847.png', '11692.png', '05609.png', '03852.png', '16079.png', '02227.png', '15781.png', '12692.png', '10768.png', '10589.png', '23677.png', '05894.png', '17813.png', '07159.png', '08976.png', '12501.png', '16177.png', '19941.png', '06427.png', '24856.png', '22618.png', '03778.png', '11508.png', '05902.png', '02974.png', '21763.png', '21790.png', '14565.png', '12625.png', '07375.png', '03762.png', '14995.png', '02156.png', '06751.png', '10654.png', '11587.png', '01241.png', '00155.png', '11917.png', '10713.png', '02743.png', '15109.png', '05912.png', '13812.png', '12621.png', '15566.png', '20484.png', '20969.png', '12442.png', '02708.png', '04192.png', '13237.png', '18681.png', '15030.png', '16918.png', '21785.png', '11296.png', '12757.png', '12159.png', '08739.png', '17832.png', '18737.png', '13656.png', '06263.png', '22090.png', '13570.png', '20048.png', '21417.png', '20020.png', '00920.png', '00043.png', '16048.png', '08294.png', '17200.png', '01676.png', '08171.png', '20632.png', '24369.png', '03943.png', '07965.png', '13332.png', '14406.png', '19242.png', '06386.png', '24630.png', '02809.png', '13167.png', '16589.png', '14597.png', '15616.png', '13912.png', '11704.png', '22770.png', '02006.png', '11936.png', '05446.png', '15245.png', '18874.png', '10829.png', '14063.png', '17712.png', '14925.png', '00989.png', '03996.png', '07223.png', '19462.png', '05319.png', '04643.png', '00255.png', '04338.png', '13208.png', '06773.png', '05740.png', '18607.png', '14300.png', '18767.png', '11485.png', '01803.png', '18027.png', '13134.png', '00588.png', '19239.png', '01996.png', '01392.png', '04777.png', '00619.png', '18530.png', '06467.png', '16032.png', '05993.png', '05230.png', '11501.png', '24647.png', '19562.png', '02205.png', '20383.png', '24625.png', '24043.png', '17710.png', '03851.png', '04771.png', '01912.png', '00438.png', '12443.png', '20364.png', '09638.png', '18387.png', '23841.png', '11736.png', '17447.png', '20947.png', '22300.png', '14453.png', '03566.png', '08615.png', '11700.png', '13517.png', '17352.png', '00762.png', '15077.png', '02479.png', '19788.png', '12389.png', '23436.png', '01492.png', '17266.png', '08458.png', '15698.png', '23725.png', '18288.png', '21275.png', '04494.png', '08578.png', '11188.png', '18244.png', '01096.png', '11670.png', '15978.png', '16333.png', '06685.png', '16890.png', '11090.png', '17371.png', '06030.png', '16974.png', '21001.png', '16886.png', '02770.png', '00333.png', '10463.png', '01463.png', '17566.png', '19537.png', '03606.png', '11308.png', '13718.png', '16486.png', '10854.png', '05046.png', '19572.png', '10930.png', '18534.png', '17979.png', '15002.png', '20183.png', '19627.png', '13256.png', '09835.png', '15298.png', '08455.png', '13149.png', '19649.png', '00932.png', '05968.png', '24809.png', '23226.png', '04910.png', '10945.png', '22529.png', '12802.png', '02240.png', '16402.png', '03200.png', '03490.png', '03028.png', '09629.png', '04550.png', '22160.png', '18327.png', '23004.png', '13102.png', '03776.png', '07315.png', '04537.png', '02184.png', '09300.png', '03554.png', '22490.png', '11971.png', '24936.png', '20787.png', '22702.png', '00360.png', '16514.png', '02797.png', '20772.png', '14448.png', '05063.png', '06515.png', '15800.png', '02739.png', '20923.png', '01624.png', '20819.png', '18778.png', '18723.png', '22929.png', '03060.png', '05303.png', '10171.png', '12499.png', '02291.png', '15050.png', '21669.png', '21532.png', '00648.png', '20638.png', '23659.png', '10029.png', '12761.png', '07731.png', '08275.png', '00775.png', '04873.png', '00384.png', '11249.png', '13190.png', '24324.png', '04545.png', '05944.png', '15671.png', '00372.png', '01234.png', '10532.png', '13306.png', '19843.png', '11429.png', '07132.png', '23358.png', '21497.png', '16084.png', '00345.png', '10918.png', '17308.png', '11446.png', '19916.png', '01061.png', '23101.png', '24680.png', '08786.png', '07432.png', '13922.png', '08058.png', '08365.png', '01172.png', '10575.png', '19085.png', '02083.png', '20384.png', '21224.png', '15890.png', '04295.png', '00716.png', '09153.png', '15744.png', '18441.png', '22200.png', '24149.png', '13776.png', '12663.png', '22738.png', '20219.png', '13015.png', '19975.png', '05823.png', '03456.png', '12506.png', '03552.png', '01368.png', '03084.png', '16104.png', '00031.png', '11474.png', '04112.png', '14896.png', '21612.png', '10334.png', '07963.png', '11643.png', '07689.png', '20214.png', '07361.png', '06334.png', '13493.png', '24948.png', '20686.png', '17063.png', '22353.png', '11523.png', '19589.png', '09849.png', '14637.png', '12876.png', '15619.png', '06529.png', '01491.png', '22851.png', '09362.png', '17802.png', '15932.png', '08511.png', '16799.png', '09418.png', '04133.png', '02569.png', '08748.png', '22357.png', '22172.png', '24451.png', '01494.png', '09013.png', '12112.png', '23671.png', '05399.png', '09670.png', '08446.png', '09859.png', '15479.png', '12930.png', '19025.png', '02696.png', '20331.png', '15329.png', '05058.png', '16394.png', '07852.png', '11585.png', '13919.png', '06001.png', '03710.png', '16388.png', '03628.png', '15042.png', '05500.png', '19324.png', '01456.png', '02837.png', '09147.png', '03020.png', '17919.png', '04806.png', '19134.png', '07369.png', '03518.png', '08558.png', '08049.png', '07952.png', '18392.png', '16057.png', '14753.png', '05348.png', '15001.png', '08140.png', '16941.png', '20880.png', '00453.png', '00467.png', '00572.png', '15972.png', '00718.png', '08717.png', '21432.png', '02495.png', '08678.png', '22531.png', '24582.png', '06925.png', '12746.png', '10186.png', '08651.png', '22814.png', '02182.png', '13352.png', '24591.png', '13057.png', '12559.png', '10151.png', '02507.png', '02357.png', '05115.png', '04275.png', '20492.png', '00184.png', '12564.png', '01987.png', '21663.png', '20752.png', '11113.png', '17645.png', '13068.png', '04791.png', '22337.png', '23935.png', '18476.png', '10265.png', '17801.png', '12994.png', '00482.png', '00089.png', '16605.png', '24471.png', '04592.png', '21828.png', '14538.png', '16126.png', '10533.png', '10191.png', '09407.png', '10995.png', '18256.png', '19727.png', '03143.png', '05237.png', '23146.png', '20976.png', '23889.png', '01428.png', '10125.png', '00178.png', '22652.png', '03088.png', '14775.png', '06491.png', '22186.png', '04328.png', '15506.png', '16132.png', '19887.png', '22069.png', '07339.png', '21573.png', '19247.png', '01379.png', '18194.png', '22361.png', '21734.png', '04029.png', '07691.png', '09329.png', '22774.png', '18899.png', '08751.png', '22747.png', '07505.png', '05841.png', '05094.png', '02528.png', '06198.png', '15281.png', '00885.png', '06280.png', '20445.png', '19512.png', '11104.png', '22598.png', '06764.png', '11015.png', '08810.png', '09712.png', '06994.png', '10808.png', '03134.png', '24589.png', '04819.png', '19191.png', '01666.png', '21448.png', '18147.png', '00577.png', '10747.png', '24691.png', '17285.png', '22441.png', '12891.png', '22963.png', '00640.png', '20713.png', '18071.png', '17849.png', '03384.png', '05772.png', '04203.png', '23061.png', '16404.png', '12077.png', '08373.png', '10439.png', '24158.png', '21042.png', '04404.png', '00891.png', '12774.png', '17569.png', '09256.png', '12881.png', '03863.png', '10144.png', '24294.png', '09104.png', '20181.png', '04104.png', '00740.png', '20604.png', '04641.png', '13909.png', '10584.png', '13313.png', '10745.png', '15297.png', '13537.png', '12819.png', '05560.png', '15581.png', '01931.png', '19442.png', '24535.png', '05866.png', '15908.png', '24015.png', '13800.png', '06294.png', '09100.png', '13132.png', '10007.png', '19008.png', '00890.png', '13499.png', '14870.png', '05434.png', '08752.png', '18135.png', '16594.png', '14967.png', '19251.png', '16508.png', '23506.png', '17067.png', '06234.png', '22588.png', '01373.png', '05056.png', '01861.png', '03066.png', '04291.png', '24587.png', '20829.png', '11279.png', '23263.png', '13104.png', '15211.png', '11384.png', '11822.png', '14932.png', '13866.png', '18059.png', '13478.png', '18404.png', '10588.png', '20267.png', '24172.png', '12869.png', '12087.png', '22419.png', '05688.png', '02978.png', '16219.png', '09656.png', '10621.png', '03662.png', '15946.png', '03497.png', '18110.png', '11800.png', '09983.png', '12647.png', '13784.png', '23953.png', '01433.png', '04763.png', '12812.png', '14604.png', '16964.png', '04108.png', '10736.png', '18100.png', '17823.png', '22449.png', '09747.png', '23652.png', '19079.png', '16977.png', '24184.png', '21045.png', '23968.png', '19158.png', '06125.png', '23113.png', '02764.png', '13046.png', '15872.png', '17461.png', '23016.png', '05622.png', '00686.png', '19712.png', '21313.png', '00738.png', '00922.png', '14490.png', '22752.png', '23798.png', '09001.png', '13213.png', '19917.png', '08097.png', '10953.png', '04686.png', '08310.png', '18075.png', '19220.png', '20620.png', '06036.png', '20222.png', '17297.png', '21711.png', '02109.png', '01510.png', '11325.png', '09560.png', '20762.png', '10512.png', '11038.png', '12178.png', '16194.png', '18240.png', '06988.png', '07459.png', '15356.png', '11867.png', '04274.png', '08978.png', '01280.png', '22771.png', '03513.png', '24215.png', '07947.png', '00892.png', '01707.png', '12270.png', '08471.png', '10122.png', '18093.png', '07753.png', '08826.png', '18471.png', '09768.png', '09998.png', '10798.png', '20616.png', '10465.png', '03639.png', '09489.png', '01083.png', '16208.png', '01152.png', '01869.png', '04520.png', '05875.png', '10156.png', '21760.png', '24560.png', '00124.png', '12572.png', '17854.png', '01632.png', '10008.png', '10286.png', '20341.png', '08325.png', '04923.png', '05748.png', '11624.png', '02124.png', '22247.png', '13287.png', '04694.png', '17270.png', '06761.png', '18151.png', '18540.png', '17705.png', '19850.png', '19640.png', '15041.png', '20293.png', '00707.png', '06331.png', '03898.png', '08759.png', '21093.png', '24913.png', '24478.png', '15758.png', '24954.png', '11914.png', '16727.png', '24178.png', '09371.png', '07690.png', '18873.png', '10418.png', '03306.png', '20922.png', '00042.png', '05714.png', '23470.png', '06345.png', '11816.png', '18805.png', '16253.png', '02699.png', '10434.png', '17795.png', '23549.png', '04084.png', '14712.png', '14282.png', '16010.png', '08799.png', '06322.png', '23544.png', '03643.png', '04541.png', '12954.png', '12842.png', '11442.png', '18267.png', '04970.png', '13926.png', '10277.png', '17760.png', '06143.png', '01395.png', '14423.png', '23834.png', '05073.png', '24880.png', '19842.png', '19366.png', '04696.png', '23703.png', '13511.png', '20647.png', '13927.png', '06848.png', '21424.png', '23169.png', '17859.png', '03926.png', '02694.png', '11377.png', '11835.png', '01682.png', '09705.png', '15825.png', '07633.png', '21843.png', '19520.png', '19272.png', '13370.png', '24102.png', '07067.png', '19604.png', '20109.png', '07606.png', '18216.png', '16331.png', '22615.png', '07412.png', '07701.png', '13687.png', '16428.png', '12682.png', '07449.png', '06201.png', '05034.png', '21118.png', '20712.png', '07984.png', '01311.png', '22498.png', '11844.png', '11220.png', '05996.png', '04650.png', '11851.png', '20035.png', '09064.png', '07567.png', '18197.png', '10550.png', '03984.png', '01450.png', '13490.png', '11101.png', '22584.png', '00924.png', '01163.png', '04216.png', '10257.png', '04449.png', '13806.png', '01625.png', '15524.png', '06887.png', '16762.png', '18700.png', '00776.png', '01628.png', '05240.png', '23145.png', '19796.png', '03136.png', '09250.png', '11043.png', '04165.png', '00598.png', '20096.png', '02144.png', '07902.png', '16071.png', '18606.png', '02363.png', '07878.png', '12219.png', '09292.png', '05561.png', '21727.png', '11196.png', '21139.png', '12337.png', '23553.png', '09321.png', '16686.png', '04591.png', '18331.png', '00824.png', '06766.png', '10716.png', '02447.png', '04188.png', '01003.png', '06244.png', '24605.png', '05357.png', '21904.png', '24031.png', '05270.png', '16170.png', '16507.png', '24285.png', '19973.png', '00741.png', '06865.png', '09948.png', '04132.png', '03968.png', '07257.png', '24652.png', '19374.png', '13682.png', '21648.png', '06747.png', '07658.png', '05564.png', '11283.png', '12727.png', '16150.png', '21144.png', '11434.png', '21557.png', '20089.png', '13863.png', '10203.png', '05108.png', '21048.png', '04748.png', '24093.png', '05278.png', '07806.png', '02976.png', '16311.png', '06533.png', '00735.png', '03099.png', '17136.png', '02365.png', '20330.png', '14053.png', '05923.png', '08891.png', '20763.png', '20525.png', '11105.png', '14960.png', '23950.png', '11663.png', '05259.png', '09608.png', '07541.png', '22125.png', '00096.png', '13637.png', '09171.png', '06755.png', '17553.png', '18246.png', '13730.png', '22175.png', '13635.png', '03465.png', '14019.png', '10795.png', '09784.png', '23797.png', '15723.png', '01629.png', '24746.png', '16574.png', '13280.png', '03168.png', '18984.png', '02840.png', '08853.png', '01901.png', '13509.png', '24729.png', '13387.png', '09632.png', '19482.png', '15557.png', '19410.png', '02389.png', '08487.png', '12380.png', '05639.png', '14222.png', '08203.png', '05678.png', '05052.png', '07823.png', '12437.png', '21910.png', '09158.png', '20352.png', '17711.png', '14233.png', '03448.png', '23970.png', '05010.png', '02981.png', '11410.png', '04091.png', '10324.png', '06669.png', '15430.png', '06216.png', '21975.png', '12970.png', '01416.png', '04047.png', '08947.png', '15576.png', '21545.png', '04533.png', '14174.png', '02738.png', '11087.png', '19279.png', '21024.png', '20188.png', '10477.png', '11471.png', '00441.png', '14271.png', '24773.png', '01710.png', '04431.png', '21733.png', '04815.png', '12469.png', '22748.png', '11393.png', '15084.png', '01313.png', '16806.png', '08969.png', '21664.png', '02150.png', '13796.png', '19315.png', '04963.png', '20071.png', '14657.png', '10763.png', '07404.png', '14779.png', '22240.png', '09766.png', '04176.png', '19370.png', '15264.png', '11687.png', '03268.png', '23479.png', '02167.png', '04237.png', '16598.png', '21216.png', '03341.png', '21450.png', '04158.png', '24234.png', '02036.png', '14054.png', '11789.png', '01356.png', '03131.png', '03843.png', '09732.png', '09690.png', '22543.png', '17877.png', '16940.png', '15446.png', '07004.png', '06623.png', '19525.png', '01070.png', '18576.png', '04867.png', '20933.png', '09109.png', '12551.png', '23099.png', '10903.png', '03125.png', '11620.png', '04538.png', '08506.png', '21902.png', '03560.png', '12581.png', '03500.png', '15570.png', '18440.png', '23171.png', '04907.png', '22162.png', '00104.png', '02836.png', '17070.png', '05468.png', '14843.png', '10775.png', '19890.png', '09548.png', '17797.png', '11630.png', '24766.png', '00295.png', '04488.png', '09425.png', '06044.png', '01262.png', '04385.png', '04452.png', '09435.png', '22679.png', '01601.png', '24852.png', '18867.png', '20546.png', '20548.png', '03816.png', '03589.png', '06223.png', '11556.png', '00436.png', '03713.png', '01094.png', '13964.png', '23973.png', '06546.png', '23045.png', '13595.png', '21356.png', '23694.png', '13488.png', '09794.png', '11306.png', '10886.png', '16169.png', '01954.png', '02756.png', '10226.png', '21191.png', '02718.png', '06653.png', '09097.png', '10717.png', '23958.png', '22143.png', '02187.png', '17602.png', '16687.png', '24679.png', '20593.png', '10800.png', '02174.png', '10561.png', '10820.png', '07209.png', '09549.png', '11258.png', '13844.png', '01971.png', '14834.png', '12862.png', '16831.png', '19105.png', '23670.png', '05702.png', '23592.png', '15528.png', '14140.png', '24700.png', '04799.png', '14651.png', '03313.png', '13399.png', '06150.png', '12043.png', '20151.png', '05022.png', '06490.png', '05991.png', '04162.png', '22862.png', '22444.png', '03218.png', '21518.png', '03602.png', '12722.png', '07724.png', '14239.png', '20360.png', '06460.png', '15831.png', '10860.png', '19347.png', '17385.png', '23904.png', '14241.png', '20513.png', '01811.png', '23164.png', '11930.png', '02131.png', '09581.png', '08684.png', '17508.png', '01690.png', '17289.png', '18876.png', '01848.png', '07099.png', '11361.png', '11075.png', '04774.png', '18200.png', '00334.png', '18541.png', '15365.png', '10382.png', '22905.png', '00262.png', '19880.png', '10376.png', '17274.png', '23135.png', '11082.png', '16001.png', '12267.png', '21853.png', '04966.png', '15886.png', '07956.png', '05457.png', '04255.png', '16435.png', '22057.png', '01879.png', '11369.png', '03370.png', '03201.png', '18481.png', '06498.png', '02028.png', '11530.png', '13614.png', '13056.png', '12286.png', '16987.png', '04749.png', '00697.png', '17572.png', '00118.png', '16915.png', '11722.png', '00552.png', '22217.png', '12830.png', '07776.png', '02427.png', '16667.png', '20653.png', '21306.png', '09154.png', '20133.png', '08624.png', '22364.png', '24259.png', '00290.png', '00087.png', '09143.png', '18610.png', '13531.png', '08917.png', '23052.png', '24179.png', '15327.png', '17316.png', '03884.png', '10970.png', '14148.png', '03378.png', '18058.png', '22243.png', '06541.png', '02027.png', '24914.png', '24186.png', '06635.png', '09589.png', '20578.png', '06555.png', '17021.png', '11438.png', '02297.png', '06948.png', '21493.png', '03349.png', '18414.png', '03479.png', '11738.png', '10944.png', '24479.png', '05871.png', '23294.png', '03075.png', '09964.png', '00650.png', '12794.png', '04231.png', '09774.png', '12131.png', '10210.png', '05472.png', '10036.png', '19597.png', '24751.png', '12222.png', '18082.png', '03123.png', '05987.png', '11951.png', '19835.png', '00628.png', '18230.png', '10019.png', '07216.png', '06338.png', '05835.png', '04945.png', '12120.png', '03972.png', '10546.png', '06301.png', '17830.png', '01720.png', '04415.png', '15183.png', '24778.png', '03712.png', '12227.png', '18731.png', '09370.png', '09348.png', '16300.png', '11649.png', '16914.png', '24332.png', '10402.png', '20173.png', '00883.png', '07906.png', '08118.png', '07345.png', '11277.png', '08733.png', '07032.png', '21735.png', '21312.png', '03701.png', '12467.png', '08637.png', '16050.png', '20158.png', '06931.png', '24400.png', '11002.png', '16588.png', '09918.png', '19595.png', '19090.png', '22118.png', '22396.png', '07272.png', '12374.png', '19267.png', '20773.png', '04486.png', '11970.png', '04554.png', '15657.png', '23724.png', '11424.png', '12010.png', '06113.png', '00206.png', '19342.png', '17575.png', '13633.png', '18423.png', '16014.png', '19235.png', '20701.png', '20858.png', '05465.png', '09598.png', '04728.png', '01727.png', '03109.png', '23695.png', '23981.png', '08749.png', '22837.png', '07766.png', '18884.png', '18024.png', '17525.png', '19922.png', '16182.png', '08687.png', '03921.png', '17791.png', '06604.png', '13058.png', '11085.png', '12945.png', '11253.png', '06284.png', '02802.png', '00717.png', '14285.png', '24919.png', '19735.png', '12382.png', '12260.png', '13720.png', '23599.png', '18081.png', '09610.png', '16639.png', '13379.png', '12449.png', '03388.png', '24097.png', '01773.png', '16963.png', '17343.png', '11246.png', '02955.png', '05839.png', '11223.png', '08951.png', '21689.png', '21672.png', '10026.png', '20239.png', '02121.png', '13289.png', '13658.png', '06205.png', '15091.png', '22761.png', '19478.png', '02278.png', '12279.png', '07705.png', '17462.png', '15311.png', '20005.png', '22852.png', '04205.png', '19657.png', '00668.png', '12711.png', '15032.png', '21096.png', '09746.png', '21795.png', '17443.png', '22049.png', '09883.png', '19173.png', '12996.png', '24352.png', '22599.png', '10224.png', '24079.png', '11124.png', '06008.png', '24777.png', '14709.png', '04115.png', '09264.png', '15593.png', '09457.png', '10021.png', '15357.png', '16622.png', '07115.png', '18989.png', '16708.png', '18165.png', '24800.png', '20406.png', '06569.png', '15034.png', '15472.png', '19533.png', '17578.png', '03894.png', '11960.png', '00995.png', '22581.png', '16449.png', '08488.png', '19020.png', '14360.png', '22806.png', '19061.png', '21125.png', '00678.png', '09919.png', '15296.png', '06962.png', '18046.png', '06034.png', '21421.png', '09877.png', '10806.png', '07267.png', '20416.png', '16792.png', '03015.png', '13628.png', '24736.png', '05315.png', '03879.png', '23462.png', '21773.png', '01975.png', '14855.png', '03419.png', '04342.png', '21680.png', '24504.png', '07398.png', '20075.png', '14659.png', '16367.png', '10227.png', '03545.png', '10072.png', '10497.png', '22605.png', '04161.png', '05925.png', '00582.png', '06811.png', '21079.png', '14673.png', '22097.png', '19452.png', '05460.png', '12630.png', '18736.png', '20122.png', '20500.png', '18549.png', '12377.png', '06296.png', '02010.png', '21140.png', '01109.png', '18511.png', '11375.png', '03708.png', '12557.png', '11365.png', '18906.png', '06924.png', '06863.png', '04680.png', '13276.png', '01060.png', '05511.png', '03126.png', '04964.png', '08495.png', '15694.png', '11381.png', '14832.png', '21778.png', '01352.png', '13694.png', '19062.png', '18783.png', '22796.png', '06503.png', '17671.png', '04069.png', '13965.png', '14663.png', '01849.png', '05558.png', '20182.png', '11708.png', '13374.png', '17870.png', '04335.png', '16488.png', '14405.png', '08643.png', '07918.png', '15079.png', '18512.png', '11177.png', '07886.png', '13756.png', '24022.png', '18461.png', '02199.png', '23377.png', '02948.png', '02870.png', '10676.png', '22189.png', '10863.png', '07422.png', '04649.png', '23381.png', '10137.png', '20281.png', '02437.png', '00781.png', '05114.png', '11437.png', '01031.png', '03034.png', '05745.png', '13758.png', '24413.png', '15044.png', '15887.png', '04534.png', '22835.png', '09328.png', '22235.png', '20861.png', '19673.png', '16151.png', '22741.png', '12939.png', '08024.png', '00705.png', '07977.png', '02334.png', '21337.png', '08776.png', '16609.png', '24938.png', '20001.png', '20820.png', '18663.png', '12635.png', '16675.png', '22574.png', '06469.png', '07864.png', '12815.png', '24723.png', '08789.png', '11566.png', '12771.png', '07088.png', '07448.png', '16781.png', '05881.png', '20887.png', '01740.png', '15496.png', '16580.png', '11301.png', '02078.png', '09709.png', '05643.png', '06692.png', '05744.png', '08596.png', '15423.png', '18319.png', '21174.png', '00362.png', '14552.png', '01729.png', '01775.png', '14050.png', '12030.png', '10129.png', '18888.png', '13666.png', '21874.png', '18031.png', '05943.png', '19415.png', '16637.png', '09349.png', '04067.png', '08297.png', '05404.png', '01748.png', '06088.png', '16730.png', '03476.png', '02471.png', '09786.png', '10782.png', '09327.png', '19088.png', '14314.png', '05884.png', '00488.png', '10168.png', '01483.png', '11343.png', '07215.png', '05587.png', '17144.png', '14383.png', '02614.png', '11193.png', '16790.png', '23134.png', '09098.png', '14517.png', '24251.png', '00973.png', '21527.png', '11305.png', '10205.png', '14636.png', '23928.png', '18497.png', '15513.png', '01725.png', '04035.png', '04909.png', '02540.png', '13232.png', '12563.png', '23761.png', '10084.png', '12570.png', '08604.png', '03938.png', '06608.png', '16452.png', '13680.png', '01795.png', '19013.png', '07264.png', '10423.png', '16434.png', '08084.png', '22773.png', '24947.png', '17411.png', '04053.png', '22902.png', '17758.png', '20256.png', '06328.png', '15930.png', '20077.png', '17826.png', '06664.png', '08230.png', '16599.png', '09038.png', '13929.png', '07186.png', '06123.png', '12549.png', '15717.png', '11062.png', '22960.png', '10131.png', '01277.png', '18225.png', '14372.png', '11924.png', '20099.png', '12098.png', '14252.png', '00400.png', '11595.png', '16982.png', '18255.png', '18577.png', '01585.png', '08718.png', '07964.png', '22685.png', '07730.png', '11781.png', '23792.png', '23458.png', '21844.png', '09676.png', '18904.png', '14371.png', '13683.png', '17096.png', '22693.png', '20098.png', '02804.png', '19868.png', '04021.png', '13650.png', '22645.png', '18427.png', '00660.png', '22370.png', '09043.png', '18976.png', '17066.png', '02464.png', '09529.png', '04947.png', '04802.png', '17810.png', '17302.png', '06245.png', '15309.png', '09476.png', '11473.png', '01960.png', '03270.png', '11049.png', '15832.png', '00456.png', '04615.png', '04233.png', '09926.png', '02972.png', '05117.png', '23778.png', '06162.png', '24296.png', '04493.png', '15150.png', '12961.png', '14150.png', '14840.png', '12042.png', '17418.png', '11273.png', '18169.png', '19667.png', '05255.png', '03287.png', '17248.png', '09662.png', '15857.png', '24748.png', '10523.png', '24775.png', '24557.png', '13825.png', '14346.png', '22001.png', '18915.png', '09602.png', '03761.png', '18376.png', '01144.png', '13979.png', '01002.png', '10149.png', '07048.png', '10845.png', '19650.png', '10900.png', '07344.png', '03954.png', '14999.png', '04542.png', '00257.png', '09971.png', '17752.png', '03705.png', '22726.png', '21787.png', '21031.png', '03582.png', '16092.png', '11868.png', '08671.png', '02221.png', '04116.png', '14502.png', '19991.png', '19146.png', '18727.png', '16772.png', '22734.png', '09982.png', '17778.png', '22246.png', '09584.png', '15643.png', '08579.png', '09394.png', '22807.png', '18789.png', '15764.png', '11328.png', '11921.png', '18715.png', '12334.png', '04637.png', '02650.png', '03571.png', '13532.png', '17139.png', '11569.png', '10720.png', '06450.png', '00376.png', '03670.png', '07337.png', '00647.png', '07265.png', '20249.png', '10221.png', '15766.png', '24789.png', '18983.png', '07648.png', '13081.png', '14234.png', '02020.png', '13305.png', '01341.png', '23838.png', '22239.png', '12992.png', '16207.png', '16960.png', '15984.png', '12324.png', '05890.png', '24536.png', '03331.png', '16067.png', '11659.png', '15747.png', '24955.png', '04221.png', '18946.png', '15958.png', '20470.png', '03752.png', '20927.png', '17727.png', '00085.png', '15716.png', '13143.png', '13195.png', '15668.png', '21856.png', '24068.png', '04622.png', '06849.png', '22998.png', '03510.png', '04382.png', '01291.png', '17166.png', '16319.png', '14656.png', '07400.png', '23081.png', '17326.png', '10245.png', '01847.png', '06391.png', '17657.png', '00402.png', '05879.png', '17018.png', '19087.png', '18380.png', '04456.png', '03990.png', '23057.png', '21768.png', '23947.png', '20019.png', '09734.png', '13634.png', '13411.png', '21560.png', '00202.png', '22336.png', '08072.png', '08897.png', '11958.png', '23389.png', '00468.png', '09586.png', '24779.png', '13376.png', '07850.png', '17704.png', '24312.png', '06471.png', '24438.png', '08224.png', '13966.png', '08221.png', '24222.png', '18509.png', '21917.png', '07086.png', '03887.png', '07311.png', '05350.png', '08610.png', '00011.png', '04168.png', '07155.png', '17456.png', '18490.png', '16309.png', '02578.png', '03262.png', '00905.png', '10392.png', '16835.png', '23151.png', '00253.png', '18893.png', '23791.png', '06173.png', '13055.png', '09036.png', '13761.png', '13291.png', '01678.png', '16465.png', '23096.png', '05415.png', '16965.png', '05789.png', '16470.png', '04574.png', '20241.png', '11968.png', '02283.png', '10250.png', '06333.png', '08194.png', '21309.png', '24208.png', '21464.png', '23736.png', '05001.png', '05011.png', '24314.png', '11658.png', '00494.png', '03583.png', '23709.png', '10333.png', '04052.png', '19744.png', '12460.png', '08892.png', '21325.png', '20354.png', '06806.png', '19425.png', '12782.png', '03797.png', '14269.png', '01693.png', '09890.png', '12251.png', '11677.png', '21124.png', '18810.png', '14197.png', '12720.png', '13318.png', '09331.png', '18535.png', '02888.png', '07617.png', '20674.png', '07675.png', '01873.png', '17992.png', '23474.png', '22890.png', '18745.png', '21402.png', '16879.png', '07602.png', '24118.png', '09654.png', '06716.png', '21151.png', '08812.png', '06941.png', '13993.png', '19430.png', '17921.png', '01699.png', '13760.png', '10856.png', '07359.png', '14575.png', '02734.png', '08794.png', '08543.png', '13075.png', '02204.png', '06969.png', '05945.png', '13454.png', '02607.png', '11628.png', '12548.png', '09711.png', '14208.png', '09051.png', '24341.png', '20012.png', '00676.png', '14293.png', '23784.png', '18850.png', '02713.png', '14615.png', '10238.png', '06836.png', '09032.png', '22644.png', '13002.png', '16114.png', '12478.png', '11680.png', '17478.png', '12108.png', '00254.png', '08936.png', '11224.png', '02253.png', '07446.png', '12558.png', '04296.png', '23106.png', '08934.png', '09388.png', '16861.png', '03920.png', '03377.png', '16034.png', '11115.png', '12789.png', '21254.png', '19320.png', '01990.png', '10396.png', '15942.png', '12793.png', '03633.png', '06352.png', '19516.png', '15301.png', '04369.png', '11045.png', '05093.png', '24940.png', '15468.png', '02823.png', '08811.png', '20296.png', '19707.png', '24320.png', '17304.png', '11653.png', '20228.png', '24903.png', '05936.png', '14297.png', '09117.png', '00116.png', '06615.png', '18963.png', '06846.png', '03379.png', '06497.png', '09102.png', '13028.png', '17685.png', '11661.png', '01829.png', '12754.png', '16447.png', '07197.png', '11792.png', '04220.png', '09237.png', '07078.png', '21749.png', '04209.png', '08972.png', '14403.png', '06051.png', '10892.png', '18096.png', '08032.png', '22881.png', '03471.png', '07169.png', '24142.png', '10472.png', '17676.png', '04473.png', '10740.png', '16082.png', '08872.png', '15268.png', '19254.png', '05335.png', '07187.png', '12562.png', '08849.png', '20090.png', '09856.png', '21514.png', '04082.png', '10424.png', '07679.png', '15323.png', '04238.png', '13851.png', '07861.png', '22969.png', '09252.png', '14292.png', '08496.png', '13842.png', '21649.png', '24872.png', '17247.png', '23153.png', '16018.png', '24960.png', '16953.png', '00873.png', '04835.png', '02655.png', '09757.png', '03895.png', '06188.png', '20580.png', '21561.png', '14007.png', '21836.png', '24845.png', '20657.png', '20742.png', '24565.png', '18385.png', '04123.png', '15455.png', '21098.png', '11359.png', '18235.png', '06630.png', '03912.png', '22848.png', '00669.png', '24362.png', '03561.png', '13631.png', '14474.png', '15719.png', '24082.png', '16027.png', '15556.png', '12063.png', '00790.png', '22046.png', '00623.png', '22791.png', '11941.png', '21719.png', '05400.png', '23060.png', '06402.png', '22354.png', '09378.png', '17783.png', '12117.png', '16395.png', '02276.png', '08167.png', '15417.png', '14818.png', '18595.png', '08165.png', '11895.png', '19051.png', '03499.png', '11864.png', '01176.png', '02508.png', '10580.png', '16962.png', '17444.png', '21485.png', '19722.png', '14436.png', '16288.png', '14693.png', '00134.png', '18709.png', '20283.png', '09825.png', '00605.png', '01177.png', '09161.png', '17690.png', '00946.png', '05104.png', '14411.png', '11517.png', '11870.png', '11810.png', '23357.png', '06980.png', '17668.png', '02662.png', '05933.png', '24350.png', '18283.png', '00974.png', '03220.png', '03398.png', '13041.png', '07893.png', '10199.png', '11229.png', '17466.png', '24546.png', '10927.png', '05719.png', '07608.png', '15080.png', '09679.png', '21816.png', '20252.png', '15835.png', '05376.png', '15637.png', '19002.png', '21761.png', '20533.png', '11466.png', '20968.png', '22045.png', '23386.png', '08740.png', '20973.png', '18220.png', '12369.png', '02208.png', '16927.png', '08894.png', '12931.png', '15604.png', '05610.png', '18179.png', '24375.png', '08028.png', '15097.png', '22438.png', '04823.png', '11528.png', '09664.png', '10400.png', '14017.png', '01642.png', '19477.png', '01493.png', '08633.png', '19834.png', '02493.png', '15413.png', '21691.png', '06970.png', '19375.png', '23124.png', '24819.png', '20262.png', '03905.png', '03309.png', '09615.png', '11702.png', '10361.png', '04175.png', '22482.png', '21726.png', '22540.png', '21693.png', '20137.png', '05479.png', '09879.png', '08701.png', '17279.png', '04628.png', '17798.png', '18026.png', '14989.png', '17945.png', '01862.png', '16832.png', '20339.png', '10276.png', '10378.png', '20661.png', '12037.png', '02008.png', '07013.png', '21883.png', '07430.png', '18945.png', '20940.png', '20990.png', '23039.png', '08295.png', '01396.png', '23339.png', '20874.png', '16053.png', '19588.png', '13413.png', '11583.png', '05458.png', '17185.png', '13865.png', '07641.png', '22469.png', '11497.png', '06448.png', '17563.png', '01102.png', '09330.png', '01184.png', '01436.png', '18643.png', '07243.png', '18770.png', '22252.png', '11042.png', '00163.png', '09473.png', '06638.png', '03651.png', '19029.png', '17700.png', '15686.png', '22585.png', '11019.png', '18871.png', '09084.png', '18144.png', '18318.png', '04934.png', '03093.png', '12058.png', '19485.png', '16108.png', '12552.png', '22716.png', '23859.png', '16956.png', '05290.png', '07625.png', '10855.png', '14781.png', '01821.png', '20846.png', '06161.png', '20456.png', '10583.png', '06734.png', '07981.png', '19521.png', '24712.png', '17454.png', '24429.png', '01257.png', '23550.png', '20509.png', '08241.png', '07727.png', '13060.png', '09620.png', '06064.png', '02828.png', '09985.png', '01486.png', '24563.png', '19326.png', '06528.png', '07173.png', '24510.png', '08336.png', '14388.png', '17073.png', '07735.png', '22834.png', '11688.png', '02573.png', '03521.png', '20981.png', '18003.png', '02015.png', '16000.png', '22035.png', '09481.png', '20195.png', '20934.png', '05781.png', '00627.png', '14686.png', '03319.png', '19481.png', '22931.png', '03176.png', '08955.png', '20872.png', '17926.png', '24049.png', '23565.png', '07844.png', '10476.png', '01709.png', '00766.png', '15096.png', '06830.png', '09515.png', '12206.png', '08254.png', '16704.png', '17681.png', '08462.png', '15503.png', '08008.png', '11400.png', '07833.png', '16120.png', '13933.png', '09604.png', '01038.png', '12354.png', '22117.png', '18506.png', '08002.png', '07523.png', '10660.png', '16929.png', '19734.png', '17736.png', '07859.png', '04797.png', '02945.png', '22478.png', '12591.png', '21425.png', '09310.png', '23560.png', '20649.png', '13111.png', '22623.png', '18550.png', '05687.png', '08047.png', '04474.png', '07089.png', '22921.png', '07841.png', '23220.png', '09539.png', '21698.png', '08000.png', '14968.png', '18474.png', '16909.png', '21454.png', '14961.png', '03417.png', '10810.png', '19394.png', '12764.png', '20343.png', '02902.png', '16839.png', '00264.png', '09953.png', '01476.png', '03002.png', '12294.png', '00714.png', '15449.png', '01296.png', '16810.png', '10728.png', '23261.png', '16621.png', '08379.png', '21037.png', '16392.png', '15222.png', '01253.png', '11829.png', '18638.png', '17483.png', '00939.png', '00538.png', '19074.png', '04857.png', '00809.png', '00495.png', '00712.png', '15344.png', '12573.png', '08688.png', '02264.png', '19140.png', '20294.png', '19330.png', '04861.png', '07407.png', '02127.png', '07742.png', '16245.png', '09995.png', '04219.png', '22298.png', '01614.png', '20682.png', '23910.png', '05419.png', '00395.png', '01895.png', '22099.png', '11155.png', '04235.png', '09915.png', '16693.png', '16838.png', '13732.png', '08478.png', '01423.png', '04624.png', '05935.png', '10586.png', '15131.png', '09372.png', '23404.png', '24431.png', '02215.png', '18314.png', '22909.png', '23003.png', '24456.png', '23322.png', '04969.png', '13088.png', '01995.png', '19045.png', '09689.png', '12976.png', '02884.png', '05570.png', '19785.png', '20464.png', '15025.png', '11794.png', '11094.png', '00789.png', '03393.png', '06091.png', '20081.png', '16623.png', '15873.png', '22329.png', '03027.png', '07066.png', '22990.png', '13807.png', '10407.png', '07058.png', '07764.png', '21376.png', '04952.png', '08863.png', '11272.png', '23378.png', '14237.png', '06659.png', '04384.png', '07015.png', '21837.png', '07292.png', '24198.png', '13460.png', '14315.png', '09607.png', '05090.png', '07519.png', '05363.png', '10455.png', '18894.png', '09054.png', '14940.png', '12224.png', '02649.png', '21747.png', '19826.png', '12803.png', '24454.png', '24898.png', '11333.png', '16145.png', '23336.png', '23366.png', '01806.png', '21179.png', '00960.png', '03935.png', '08875.png', '01080.png', '07605.png', '19647.png', '22499.png', '09142.png', '00814.png', '17911.png', '15289.png', '10562.png', '02909.png', '21764.png', '21039.png', '05616.png', '23405.png', '03526.png', '02459.png', '09212.png', '05402.png', '00055.png', '15549.png', '11623.png', '15690.png', '22464.png', '06062.png', '03699.png', '20087.png', '19736.png', '19701.png', '20076.png', '06032.png', '08392.png', '14927.png', '21657.png', '23629.png', '16942.png', '20466.png', '23293.png', '23009.png', '04789.png', '11475.png', '19417.png', '18716.png', '00519.png', '02530.png', '15444.png', '22829.png', '15171.png', '03861.png', '01032.png', '06154.png', '21594.png', '19856.png', '17793.png', '21993.png', '12547.png', '18817.png', '08370.png', '15552.png', '00186.png', '09509.png', '08430.png', '18367.png', '02207.png', '12882.png', '19690.png', '04460.png', '02758.png', '21861.png', '07001.png', '13498.png', '09434.png', '03735.png', '08540.png', '24799.png', '02781.png', '09469.png', '21478.png', '00275.png', '10201.png', '04232.png', '04257.png', '05453.png', '08491.png', '21706.png', '17646.png', '16322.png', '16242.png', '21804.png', '03427.png', '17634.png', '07856.png', '03779.png', '22420.png', '10501.png', '04691.png', '17056.png', '23554.png', '01796.png', '22501.png', '15138.png', '08305.png', '01405.png', '13872.png', '00489.png', '16769.png', '18777.png', '07930.png', '24270.png', '00688.png', '17190.png', '06419.png', '15626.png', '14638.png', '06207.png', '10158.png', '18785.png', '11138.png', '18800.png', '04057.png', '14236.png', '19409.png', '23897.png', '10590.png', '20970.png', '21490.png', '13673.png', '15304.png', '13915.png', '05264.png', '08595.png', '19211.png', '19498.png', '16094.png', '15935.png', '00561.png', '00881.png', '11633.png', '11918.png', '08222.png', '03636.png', '07560.png', '16501.png', '18551.png', '16090.png', '02266.png', '15340.png', '23559.png', '24961.png', '04289.png', '04215.png', '11896.png', '00164.png', '14077.png', '10613.png', '00175.png', '12303.png', '08371.png', '19422.png', '10815.png', '22258.png', '23990.png', '02633.png', '10087.png', '08944.png', '23732.png', '07873.png', '15031.png', '07528.png', '15896.png', '18959.png', '06191.png', '23066.png', '01307.png', '23507.png', '04567.png', '23087.png', '20951.png', '08320.png', '15792.png', '00615.png', '24944.png', '18616.png', '10540.png', '16528.png', '13216.png', '02195.png', '16489.png', '18706.png', '20039.png', '17547.png', '21014.png', '09795.png', '03157.png', '21199.png', '15816.png', '15107.png', '02371.png', '04635.png', '02477.png', '11565.png', '22164.png', '18432.png', '24722.png', '01042.png', '10704.png', '18188.png', '01324.png', '19285.png', '09150.png', '14011.png', '10539.png', '22910.png', '09134.png', '04218.png', '22970.png', '13592.png', '11154.png', '08627.png', '09531.png', '19217.png', '04249.png', '24293.png', '18371.png', '20963.png', '21989.png', '17946.png', '14492.png', '24658.png', '17599.png', '22888.png', '22992.png', '13073.png', '11012.png', '09012.png', '16399.png', '17251.png', '02001.png', '13824.png', '05897.png', '03042.png', '17576.png', '22411.png', '04248.png', '04223.png', '05620.png', '16065.png', '11459.png', '20250.png', '02686.png', '04928.png', '18362.png', '23683.png', '17299.png', '03610.png', '02439.png', '09655.png', '07348.png', '21576.png', '24757.png', '01543.png', '23409.png', '20327.png', '20376.png', '16600.png', '19047.png', '12968.png', '13828.png', '12173.png', '21901.png', '20318.png', '18155.png', '01539.png', '04981.png', '00645.png', '00368.png', '03531.png', '14477.png', '06507.png', '02678.png', '00119.png', '24323.png', '12333.png', '06730.png', '21936.png', '17204.png', '10491.png', '00595.png', '15408.png', '14418.png', '14876.png', '19893.png', '22407.png', '12049.png', '18773.png', '06070.png', '22559.png', '06712.png', '07389.png', '11825.png', '20897.png', '16919.png', '04327.png', '17265.png', '24611.png', '20407.png', '12290.png', '24127.png', '13995.png', '17013.png', '17557.png', '14052.png', '11608.png', '03901.png', '14614.png', '06736.png', '07064.png', '04114.png', '01615.png', '22589.png', '02677.png', '10946.png', '24424.png', '24151.png', '04921.png', '14396.png', '05323.png', '08941.png', '18656.png', '05286.png', '01281.png', '22165.png', '06277.png', '00210.png', '11479.png', '06179.png', '19706.png', '01586.png', '10060.png', '08673.png', '19120.png', '23740.png', '08173.png', '23424.png', '13312.png', '09820.png', '18692.png', '19583.png', '05057.png', '08334.png', '02213.png', '12415.png', '11418.png', '06219.png', '24132.png', '24917.png', '17187.png', '03437.png', '05995.png', '11152.png', '11714.png', '05856.png', '16943.png', '13809.png', '01108.png', '09442.png', '23708.png', '22959.png', '20100.png', '13560.png', '06133.png', '03435.png', '22831.png', '06063.png', '21053.png', '11023.png', '13230.png', '22296.png', '12153.png', '07008.png', '20937.png', '10987.png', '08613.png', '13688.png', '21426.png', '12452.png', '02152.png', '07745.png', '17535.png', '08363.png', '19044.png', '04661.png', '14401.png', '02960.png', '11777.png', '23063.png', '06451.png', '13868.png', '16244.png', '00466.png', '02627.png', '22475.png', '04792.png', '05142.png', '09111.png', '09258.png', '16729.png', '14460.png', '01067.png', '17971.png', '01115.png', '19723.png', '21933.png', '17008.png', '12083.png', '18940.png', '06687.png', '14935.png', '24440.png', '23530.png', '14498.png', '23905.png', '08301.png', '11542.png', '14677.png', '00142.png', '12801.png', '14894.png', '01863.png', '16357.png', '14048.png', '02369.png', '11185.png', '04421.png', '05573.png', '11521.png', '23437.png', '08706.png', '21108.png', '01314.png', '18431.png', '14983.png', '15406.png', '16328.png', '03412.png', '24159.png', '12352.png', '14367.png', '04679.png', '09749.png', '13204.png', '11887.png', '03519.png', '07959.png', '21911.png', '20036.png', '21941.png', '21655.png', '23094.png', '07402.png', '20032.png', '18840.png', '10932.png', '09493.png', '17929.png', '02327.png', '19041.png', '12346.png', '23927.png', '17499.png', '12989.png', '16334.png', '22790.png', '22692.png', '09503.png', '21885.png', '13464.png', '10556.png', '22285.png', '05490.png', '23742.png', '12149.png', '24678.png', '22365.png', '10350.png', '21488.png', '17567.png', '10678.png', '18104.png', '07338.png', '24709.png', '03540.png', '13575.png', '01756.png', '16037.png', '02032.png', '00610.png', '14788.png', '08233.png', '15072.png', '14079.png', '02936.png', '05765.png', '22376.png', '08007.png', '10513.png', '13604.png', '00551.png', '06877.png', '17163.png', '12404.png', '15461.png', '00381.png', '19987.png', '08542.png', '21817.png', '06152.png', '15437.png', '10626.png', '19111.png', '20216.png', '21623.png', '00358.png', '22675.png', '23980.png', '01403.png', '24802.png', '07068.png', '17288.png', '11251.png', '19762.png', '10822.png', '07160.png', '23847.png', '17928.png', '21315.png', '09197.png', '11747.png', '15494.png', '08158.png', '06860.png', '00902.png', '16509.png', '21177.png', '06684.png', '20129.png', '11302.png', '17243.png', '22290.png', '01658.png', '20685.png', '08919.png', '03961.png', '04858.png', '00356.png', '06086.png', '09181.png', '24764.png', '18486.png', '05588.png', '13506.png', '01254.png', '15509.png', '16184.png', '18679.png', '11769.png', '21621.png', '05664.png', '17897.png', '02744.png', '20557.png', '05805.png', '02531.png', '13476.png', '17375.png', '08994.png', '11401.png', '16787.png', '01843.png', '11494.png', '07640.png', '22968.png', '05279.png', '10457.png', '08110.png', '05770.png', '23697.png', '01103.png', '12607.png', '24503.png', '04472.png', '08592.png', '19371.png', '18722.png', '14811.png', '12873.png', '07062.png', '20919.png', '05019.png', '23890.png', '16042.png', '02588.png', '17392.png', '05931.png', '07697.png', '16152.png', '11524.png', '18366.png', '24169.png', '03893.png', '17284.png', '06033.png', '18473.png', '12657.png', '11817.png', '18342.png', '08094.png', '11657.png', '14933.png', '08713.png', '12004.png', '13011.png', '18857.png', '10410.png', '23036.png', '03361.png', '23913.png', '03386.png', '12391.png', '19671.png', '03451.png', '15146.png', '10015.png', '07923.png', '05167.png', '10233.png', '03791.png', '10239.png', '01063.png', '00919.png', '10729.png', '15839.png', '14868.png', '21652.png', '15656.png', '06835.png', '17306.png', '21279.png', '19459.png', '08042.png', '07326.png', '12927.png', '17442.png', '07525.png', '05774.png', '15925.png', '09379.png', '18353.png', '10196.png', '09763.png', '15559.png', '06511.png', '13462.png', '17900.png', '00753.png', '17862.png', '05635.png', '14815.png', '09062.png', '14305.png', '08029.png', '08009.png', '08932.png', '18419.png', '05036.png', '21030.png', '14547.png', '01189.png', '18479.png', '24702.png', '06654.png', '18253.png', '21855.png', '13953.png', '12866.png', '24152.png', '02474.png', '23461.png', '16256.png', '10883.png', '14124.png', '16422.png', '20023.png', '07108.png', '17863.png', '07000.png', '07769.png', '01171.png', '22882.png', '09413.png', '22680.png', '06707.png', '10217.png', '19054.png', '08404.png', '21210.png', '16091.png', '14536.png', '23896.png', '23611.png', '01712.png', '22513.png', '13555.png', '05764.png', '24682.png', '01694.png', '04898.png', '17850.png', '06131.png', '10762.png', '09071.png', '19945.png', '21354.png', '17222.png', '21299.png', '16665.png', '06196.png', '12214.png', '21362.png', '07034.png', '05867.png', '22387.png', '23338.png', '03390.png', '09630.png', '23676.png', '14368.png', '05532.png', '16754.png', '04462.png', '13064.png', '16444.png', '07831.png', '10647.png', '03919.png', '03660.png', '09354.png', '22535.png', '11035.png', '14126.png', '12300.png', '01116.png', '02584.png', '21770.png', '15634.png', '18400.png', '09011.png', '14990.png', '02404.png', '20621.png', '16524.png', '05598.png', '21272.png', '23242.png', '12566.png', '10681.png', '04878.png', '00346.png', '14384.png', '07876.png', '09755.png', '02737.png', '06800.png', '16549.png', '01844.png', '13443.png', '02435.png', '05885.png', '17330.png', '18933.png', '03854.png', '15144.png', '24844.png', '02138.png', '09016.png', '15453.png', '16254.png', '17608.png', '07259.png', '10950.png', '19848.png', '17536.png', '00825.png', '05499.png', '11289.png', '23644.png', '00277.png', '00220.png', '12156.png', '13009.png', '20907.png', '04985.png', '11602.png', '08660.png', '16676.png', '02351.png', '19170.png', '07848.png', '03909.png', '23216.png', '24804.png', '07145.png', '03697.png', '20617.png', '21779.png', '06705.png', '10024.png', '16610.png', '17933.png', '13418.png', '18910.png', '20276.png', '20395.png', '02171.png', '11392.png', '09342.png', '14593.png', '12204.png', '21758.png', '01708.png', '13513.png', '01029.png', '21041.png', '05210.png', '23355.png', '00245.png', '06834.png', '01860.png', '00984.png', '23814.png', '09583.png', '20530.png', '05381.png', '16371.png', '12688.png', '05343.png', '19891.png', '09268.png', '05241.png', '20573.png', '06946.png', '04434.png', '01948.png', '17201.png', '19672.png', '14508.png', '16786.png', '18099.png', '18304.png', '11638.png', '01122.png', '21435.png', '19653.png', '24505.png', '03026.png', '14231.png', '13226.png', '03881.png', '12487.png', '16887.png', '17515.png', '17426.png', '24442.png', '19892.png', '09092.png', '12220.png', '03474.png', '00269.png', '23587.png', '10273.png', '17494.png', '08438.png', '08895.png', '14399.png', '15809.png', '00683.png', '19587.png', '20067.png', '24549.png', '10177.png', '17871.png', '11280.png', '03718.png', '02049.png', '23509.png', '23583.png', '20605.png', '17675.png', '16567.png', '15055.png', '15247.png', '16353.png', '20588.png', '10598.png', '00499.png', '17665.png', '09792.png', '09839.png', '19022.png', '06517.png', '05026.png', '18324.png', '18345.png', '06551.png', '05713.png', '08628.png', '21774.png', '21984.png', '06881.png', '23465.png', '18291.png', '07424.png', '21854.png', '06738.png', '03071.png', '02396.png', '24883.png', '20234.png', '13330.png', '18592.png', '07837.png', '12836.png', '14752.png', '06306.png', '19551.png', '18555.png', '04442.png', '21686.png', '06464.png', '15859.png', '05873.png', '12170.png', '19818.png', '23174.png', '08537.png', '09692.png', '22272.png', '03120.png', '04828.png', '02202.png', '08572.png', '08016.png', '17667.png', '12696.png', '02593.png', '14123.png', '09116.png', '16995.png', '00497.png', '09697.png', '19616.png', '15161.png', '16720.png', '04339.png', '02952.png', '00211.png', '08656.png', '04476.png', '19513.png', '12021.png', '18206.png', '23839.png', '13738.png', '17661.png', '13155.png', '06770.png', '16266.png', '21437.png', '02516.png', '03859.png', '00911.png', '17903.png', '24545.png', '14737.png', '00037.png', '08298.png', '09438.png', '13468.png', '03255.png', '21422.png', '10302.png', '05166.png', '06649.png', '23150.png', '10179.png', '06721.png', '14442.png', '08530.png', '08835.png', '23588.png', '10813.png', '15338.png', '22406.png', '11282.png', '01447.png', '18221.png', '12863.png', '12693.png', '03016.png', '00684.png', '11027.png', '18994.png', '21504.png', '17639.png', '07813.png', '05509.png', '04901.png', '03475.png', '01732.png', '01633.png', '13271.png', '17772.png', '08197.png', '20121.png', '24055.png', '06342.png', '09729.png', '18978.png', '18820.png', '04709.png', '08020.png', '22224.png', '07556.png', '04732.png', '19069.png', '10922.png', '06906.png', '22067.png', '06719.png', '12181.png', '21938.png', '09941.png', '19932.png', '24271.png', '03433.png', '16012.png', '05314.png', '12124.png', '09500.png', '08979.png', '08343.png', '16262.png', '20057.png', '05872.png', '17397.png', '08410.png', '10268.png', '09847.png', '00265.png', '24867.png', '17530.png', '07797.png', '06355.png', '22053.png', '21172.png', '16681.png', '05091.png', '22007.png', '16460.png', '15345.png', '02339.png', '12775.png', '19050.png', '21120.png', '21709.png', '15866.png', '16264.png', '11696.png', '06493.png', '03234.png', '11443.png', '14324.png', '15693.png', '15145.png', '11382.png', '00625.png', '03642.png', '20306.png', '23636.png', '10666.png', '11982.png', '11859.png', '15630.png', '24596.png', '00522.png', '08939.png', '06041.png', '22949.png', '15005.png', '11163.png', '18348.png', '07653.png', '04290.png', '18307.png', '03429.png', '03346.png', '24529.png', '07874.png', '00271.png', '16200.png', '19774.png', '00876.png', '03243.png', '15062.png', '14205.png', '24633.png', '24500.png', '08981.png', '01030.png', '17448.png', '09080.png', '24595.png', '15910.png', '11447.png', '00008.png', '04239.png', '19209.png', '14310.png', '14505.png', '24177.png', '11391.png', '12408.png', '06940.png', '01555.png', '04358.png', '16345.png', '09923.png', '20372.png', '18517.png', '05634.png', '09316.png', '17292.png', '12751.png', '16068.png', '18951.png', '12261.png', '10751.png', '22176.png', '05668.png', '03387.png', '04931.png', '03668.png', '16780.png', '04729.png', '07559.png', '08445.png', '15475.png', '03569.png', '12405.png', '11431.png', '00343.png', '03239.png', '14270.png', '18057.png', '02185.png', '14695.png', '02805.png', '20805.png', '22580.png', '12132.png', '04924.png', '22294.png', '00010.png', '21203.png', '02856.png', '10241.png', '22492.png', '13947.png', '01186.png', '00599.png', '13383.png', '15560.png', '20730.png', '12999.png', '11849.png', '09954.png', '20757.png', '06845.png', '19204.png', '12061.png', '00726.png', '11489.png', '11691.png', '02149.png', '00464.png', '15132.png', '12778.png', '02325.png', '19071.png', '00120.png', '11102.png', '22259.png', '16748.png', '17731.png', '05552.png', '07618.png', '14992.png', '21818.png', '11200.png', '07763.png', '17489.png', '15038.png', '07319.png', '03272.png', '08993.png', '00063.png', '01536.png', '01758.png', '06674.png', '20483.png', '07827.png', '09554.png', '04093.png', '14356.png', '07479.png', '18485.png', '16378.png', '04657.png', '09635.png', '01193.png', '09067.png', '06212.png', '03870.png', '06965.png', '16222.png', '12448.png', '01093.png', '17973.png', '04304.png', '09263.png', '02194.png', '01419.png', '15358.png', '17071.png', '13667.png', '10582.png', '10529.png', '10895.png', '07336.png', '12199.png', '13905.png', '17450.png', '00232.png', '18341.png', '05031.png', '05747.png', '13702.png', '11151.png', '18269.png', '07588.png', '11900.png', '10976.png', '19496.png', '23532.png', '00419.png', '06230.png', '11192.png', '24791.png', '19506.png', '01131.png', '21813.png', '15732.png', '24287.png', '07795.png', '00593.png', '22467.png', '24585.png', '21119.png', '18578.png', '13067.png', '14309.png', '08234.png', '16409.png', '13026.png', '13523.png', '12353.png', '03514.png', '14013.png', '00586.png', '13582.png', '21453.png', '08999.png', '10331.png', '06372.png', '05611.png', '11642.png', '10552.png', '13294.png', '19365.png', '23068.png', '18631.png', '06381.png', '20021.png', '08337.png', '00694.png', '20051.png', '05623.png', '16064.png', '19703.png', '12694.png', '10557.png', '01854.png', '18735.png', '13714.png', '21889.png', '03468.png', '06616.png', '23504.png', '10817.png', '06178.png', '09587.png', '09003.png', '03271.png', '22483.png', '10353.png', '21677.png', '22638.png', '21101.png', '21409.png', '05071.png', '06991.png', '01251.png', '06534.png', '18219.png', '16830.png', '04280.png', '20790.png', '03782.png', '20110.png', '20804.png', '06958.png', '24490.png', '22494.png', '00742.png', '01679.png', '24942.png', '11543.png', '06308.png', '16678.png', '20263.png', '08014.png', '08102.png', '24586.png', '16391.png', '10344.png', '01559.png', '22072.png', '13196.png', '23203.png', '00917.png', '08497.png', '23335.png', '04778.png', '09135.png', '17146.png', '11772.png', '15143.png', '14719.png', '01856.png', '22887.png', '22799.png', '10252.png', '21918.png', '11997.png', '16945.png', '06370.png', '22205.png', '11449.png', '03630.png', '10878.png', '12401.png', '11172.png', '22772.png', '06737.png', '17432.png', '24122.png', '19232.png', '08912.png', '22730.png', '22348.png', '08041.png', '13471.png', '21908.png', '21192.png', '12490.png', '01898.png', '02397.png', '07714.png', '23872.png', '15003.png', '14689.png', '12238.png', '22539.png', '24200.png', '16619.png', '00357.png', '15013.png', '00815.png', '11467.png', '08631.png', '05663.png', '10253.png', '01235.png', '19077.png', '21214.png', '23582.png', '05847.png', '15767.png', '23975.png', '15409.png', '20165.png', '10093.png', '24684.png', '14758.png', '12833.png', '13103.png', '02757.png', '19303.png', '23431.png', '12197.png', '20893.png', '21644.png', '01074.png', '24820.png', '06735.png', '15390.png', '24115.png', '09377.png', '11055.png', '19586.png', '19018.png', '01145.png', '13786.png', '24622.png', '04810.png', '19360.png', '09781.png', '08355.png', '22270.png', '24821.png', '15894.png', '02913.png', '20482.png', '08676.png', '06897.png', '14041.png', '04086.png', '00696.png', '11770.png', '05858.png', '19099.png', '06960.png', '04336.png', '14624.png', '19165.png', '02087.png', '15736.png', '08609.png', '14826.png', '14532.png', '12032.png', '05682.png', '09730.png', '11131.png', '20111.png', '18623.png', '04356.png', '15869.png', '23566.png', '06378.png', '05134.png', '16912.png', '16573.png', '12740.png', '09840.png', '01479.png', '14061.png', '08125.png', '18883.png', '20135.png', '00187.png', '15595.png', '11044.png', '01661.png', '00132.png', '07629.png', '21786.png', '15458.png', '14347.png', '19863.png', '22527.png', '14326.png', '06814.png', '14320.png', '20491.png', '03769.png', '00961.png', '23224.png', '10098.png', '02051.png', '22592.png', '06398.png', '18228.png', '12007.png', '22091.png', '14860.png', '00078.png', '01986.png', '07475.png', '07354.png', '11171.png', '07661.png', '02410.png', '13181.png', '04958.png', '13651.png', '01359.png', '21410.png', '06726.png', '15874.png', '22421.png', '02075.png', '07244.png', '15745.png', '21300.png', '01953.png', '00028.png', '24273.png', '14471.png', '08212.png', '12371.png', '15037.png', '12311.png', '09967.png', '08906.png', '12700.png', '15941.png', '07539.png', '16745.png', '02861.png', '05892.png', '01837.png', '23924.png', '04875.png', '08440.png', '02286.png', '22341.png', '00562.png', '22015.png', '24475.png', '01289.png', '23867.png', '24433.png', '23144.png', '08259.png', '21538.png', '16240.png', '13918.png', '20777.png', '11600.png', '11137.png', '16629.png', '24340.png', '06787.png', '01552.png', '02866.png', '21308.png', '20744.png', '24538.png', '09273.png', '15397.png', '20504.png', '06189.png', '23247.png', '01818.png', '17412.png', '09160.png', '15773.png', '10984.png', '17170.png', '03191.png', '03991.png', '07834.png', '15166.png', '13613.png', '10445.png', '03359.png', '16110.png', '21301.png', '13936.png', '01391.png', '18433.png', '24813.png', '22976.png', '14857.png', '16236.png', '00224.png', '00112.png', '19108.png', '19839.png', '01897.png', '13120.png', '06525.png', '06035.png', '21081.png', '22749.png', '10189.png', '17937.png', '15442.png', '13679.png', '03765.png', '19451.png', '21606.png', '03353.png', '05734.png', '08722.png', '14772.png', '19515.png', '12542.png', '22023.png', '19312.png', '06085.png', '12400.png', '18261.png', '14369.png', '23141.png', '13654.png', '22522.png', '12407.png', '08606.png', '02877.png', '15362.png', '16393.png', '01595.png', '15923.png', '16325.png', '04765.png', '18992.png', '24615.png', '22433.png', '10639.png', '05072.png', '09603.png', '23519.png', '03224.png', '21032.png', '15917.png', '01353.png', '21347.png', '17679.png', '04496.png', '17583.png', '15899.png', '01449.png', '00875.png', '01575.png', '20002.png', '00526.png', '20215.png', '23419.png', '12309.png', '16526.png', '07816.png', '02044.png', '16190.png', '17125.png', '12223.png', '03645.png', '16642.png', '17618.png', '04960.png', '14880.png', '05483.png', '20785.png', '02373.png', '02676.png', '18591.png', '03562.png', '24346.png', '15947.png', '01551.png', '17869.png', '12165.png', '09640.png', '03573.png', '16880.png', '11957.png', '17624.png', '13749.png', '10748.png', '14740.png', '19156.png', '15762.png', '09179.png', '22063.png', '21702.png', '11866.png', '12488.png', '11074.png', '05275.png', '08605.png', '17464.png', '09287.png', '10296.png', '14349.png', '03113.png', '17959.png', '03707.png', '20722.png', '19104.png', '10033.png', '08639.png', '21061.png', '06080.png', '13726.png', '04281.png', '19607.png', '11388.png', '24855.png', '06136.png', '19579.png', '15284.png', '04311.png', '20139.png', '16023.png', '11003.png', '24640.png', '19083.png', '00147.png', '00095.png', '22624.png', '12755.png', '05795.png', '21420.png', '05386.png', '22724.png', '24508.png', '06110.png', '24372.png', '17102.png', '12376.png', '08711.png', '01462.png', '11239.png', '09751.png', '24594.png', '05694.png', '24846.png', '01166.png', '04260.png', '10454.png', '22128.png', '18502.png', '04972.png', '15836.png', '07096.png', '02982.png', '18902.png', '00005.png', '04345.png', '15968.png', '01626.png', '19400.png', '10577.png', '00788.png', '12912.png', '13045.png', '10974.png', '24289.png', '02827.png', '24444.png', '06314.png', '03030.png', '16701.png', '11773.png', '11080.png', '07996.png', '09439.png', '20309.png', '09921.png', '09360.png', '20589.png', '16408.png', '19851.png', '18627.png', '04245.png', '05766.png', '20136.png', '00437.png', '08500.png', '20223.png', '11707.png', '00412.png', '00111.png', '12041.png', '15961.png', '12864.png', '00886.png', '20736.png', '20286.png', '01606.png', '09527.png', '12068.png', '15307.png', '17915.png', '22241.png', '16421.png', '09910.png', '09455.png', '15330.png', '09141.png', '23983.png', '22765.png', '01269.png', '10821.png', '21381.png', '11363.png', '09357.png', '04852.png', '14221.png', '01192.png', '00980.png', '09339.png', '21550.png', '15960.png', '03055.png', '02682.png', '15772.png', '09050.png', '12232.png', '12786.png', '19448.png', '14454.png', '03855.png', '13719.png', '06374.png', '14598.png', '09986.png', '05251.png', '13245.png', '13963.png', '02481.png', '22231.png', '02872.png', '23002.png', '01054.png', '12969.png', '06163.png', '00757.png', '11164.png', '04149.png', '23493.png', '00252.png', '17622.png', '16080.png', '22147.png', '00289.png', '17314.png', '18675.png', '22253.png', '00545.png', '23376.png', '09277.png', '22393.png', '21065.png', '00054.png', '06904.png', '24878.png', '01865.png', '24361.png', '11535.png', '09467.png', '02165.png', '10801.png', '16180.png', '11120.png', '15936.png', '15749.png', '22884.png', '22608.png', '13891.png', '15026.png', '19957.png', '02775.png', '09641.png', '22264.png', '14628.png', '11808.png', '09944.png', '10289.png', '02216.png', '08938.png', '12217.png', '00811.png', '04180.png', '01872.png', '21236.png', '06531.png', '07431.png', '07316.png', '05048.png', '06997.png', '16989.png', '10643.png', '15730.png', '03210.png', '02198.png', '10004.png', '19042.png', '16228.png', '17724.png', '15687.png', '04465.png', '15443.png', '02132.png', '15905.png', '13960.png', '14495.png', '21020.png', '04736.png', '06357.png', '10043.png', '18112.png', '04080.png', '05445.png', '10234.png', '09613.png', '03013.png', '20630.png', '01223.png', '20350.png', '07495.png', '15070.png', '07501.png', '03395.png', '04032.png', '10993.png', '12808.png', '01917.png', '12914.png', '15341.png', '01565.png', '18463.png', '09884.png', '04113.png', '10130.png', '09183.png', '13351.png', '10658.png', '22195.png', '14125.png', '17391.png', '06274.png', '15000.png', '18954.png', '15770.png', '23466.png', '17834.png', '18756.png', '15802.png', '23633.png', '02926.png', '09873.png', '16380.png', '24871.png', '20667.png', '19747.png', '01908.png', '08935.png', '18469.png', '04025.png', '03844.png', '23243.png', '16932.png', '05771.png', '14005.png', '16384.png', '16563.png', '02319.png', '16910.png', '13119.png', '10055.png', '23044.png', '08559.png', '22339.png', '06701.png', '10719.png', '11567.png', '11286.png', '24075.png', '16742.png', '01656.png', '22981.png', '13873.png', '05207.png', '08059.png', '14201.png', '06763.png', '14761.png', '21507.png', '04914.png', '13438.png', '15303.png', '23971.png', '08590.png', '01524.png', '08472.png', '20553.png', '19937.png', '02040.png', '02973.png', '14692.png', '18546.png', '04626.png', '11954.png', '09417.png', '15534.png', '06954.png', '01438.png', '00377.png', '22758.png', '04501.png', '10441.png', '05372.png', '14240.png', '06435.png', '12679.png', '19921.png', '11472.png', '03316.png', '09094.png', '23668.png', '11737.png', '23777.png', '15883.png', '23806.png', '09400.png', '16480.png', '16493.png', '06825.png', '21678.png', '12356.png', '04734.png', '11665.png', '23408.png', '17845.png', '09845.png', '08570.png', '06740.png', '05398.png', '17140.png', '04041.png', '08707.png', '16154.png', '14478.png', '20301.png', '12174.png', '21472.png', '09222.png', '21091.png', '13247.png', '21745.png', '11742.png', '20244.png', '03685.png', '13587.png', '19621.png', '19878.png', '07949.png', '00529.png', '12648.png', '04464.png', '00900.png', '14026.png', '10783.png', '18543.png', '08126.png', '14317.png', '03166.png', '13158.png', '22077.png', '00370.png', '00693.png', '05640.png', '14592.png', '19968.png', '17089.png', '17471.png', '24176.png', '22961.png', '00525.png', '02631.png', '18604.png', '21182.png', '23612.png', '11162.png', '01265.png', '14020.png', '03724.png', '05969.png', '00320.png', '21205.png', '24927.png', '18570.png', '19634.png', '23537.png', '09140.png', '03091.png', '21353.png', '05988.png', '02705.png', '09864.png', '15618.png', '10209.png', '18565.png', '14591.png', '05820.png', '10279.png', '12683.png', '13748.png', '07517.png', '17855.png', '07962.png', '15076.png', '24716.png', '17975.png', '13957.png', '04334.png', '13972.png', '18956.png', '20679.png', '13378.png', '03934.png', '02299.png', '10770.png', '04007.png', '11513.png', '08834.png', '16819.png', '08437.png', '14823.png', '22373.png', '17242.png', '12183.png', '06789.png', '17647.png', '18912.png', '17191.png', '19531.png', '04699.png', '22093.png', '23501.png', '15420.png', '11762.png', '11952.png', '03924.png', '04489.png', '05308.png', '01160.png', '21455.png', '19558.png', '24244.png', '01169.png', '21608.png', '24772.png', '01581.png', '14564.png', '01887.png', '22754.png', '06192.png', '17130.png', '00185.png', '15733.png', '21924.png', '01037.png', '10818.png', '10623.png', '01295.png', '12235.png', '18771.png', '15293.png', '04417.png', '24398.png', '04558.png', '00869.png', '19390.png', '02261.png', '17885.png', '10636.png', '15856.png', '23446.png', '06805.png', '08709.png', '02543.png', '21388.png', '23840.png', '06596.png', '03347.png', '21519.png', '20429.png', '19164.png', '19906.png', '03186.png', '16603.png', '06175.png', '14429.png', '18921.png', '01465.png', '03117.png', '17457.png', '23330.png', '17851.png', '04044.png', '08793.png', '03806.png', '09903.png', '00502.png', '24732.png', '22986.png', '24891.png', '19763.png', '06900.png', '04879.png', '10734.png', '16767.png', '04716.png', '23304.png', '21378.png', '09251.png', '22473.png', '21416.png', '16302.png', '18765.png', '24219.png', '17718.png', '12189.png', '20741.png', '17556.png', '18190.png', '23597.png', '08315.png', '04673.png', '20381.png', '09957.png', '12901.png', '16673.png', '05597.png', '05471.png', '21699.png', '19389.png', '17012.png', '07016.png', '15262.png', '08877.png', '08304.png', '10904.png', '00153.png', '20738.png', '03736.png', '05337.png', '01769.png', '20498.png', '17369.png', '11098.png', '13348.png', '10700.png', '06177.png', '13871.png', '00653.png', '19219.png', '14976.png', '09819.png', '16503.png', '05312.png', '01338.png', '19688.png', '07576.png', '07699.png', '17356.png', '19864.png', '17211.png', '18406.png', '01819.png', '07439.png', '01783.png', '24656.png', '11309.png', '06452.png', '14713.png', '22803.png', '01442.png', '15828.png', '11355.png', '13987.png', '07761.png', '24674.png', '10335.png', '17458.png', '20584.png', '21636.png', '10405.png', '11054.png', '03144.png', '23325.png', '19484.png', '12109.png', '24752.png', '23031.png', '19392.png', '17438.png', '07214.png', '19421.png', '20939.png', '02383.png', '21890.png', '18455.png', '12093.png', '13047.png', '20488.png', '10737.png', '04458.png', '04747.png', '03274.png', '09144.png', '15639.png', '02455.png', '23258.png', '08439.png', '03741.png', '14819.png', '10267.png', '19633.png', '01896.png', '21134.png', '08121.png', '03194.png', '22032.png', '13078.png', '04079.png', '17337.png', '08144.png', '11336.png', '22436.png', '01227.png', '11122.png', '20766.png', '11181.png', '12526.png', '14157.png', '15090.png', '08476.png', '06237.png', '18077.png', '01263.png', '06592.png', '22254.png', '01175.png', '11959.png', '11650.png', '08442.png', '00756.png', '17003.png', '04346.png', '03889.png', '07511.png', '22352.png', '16540.png', '07082.png', '10605.png', '14414.png', '05864.png', '06744.png', '03734.png', '00256.png', '19327.png', '14734.png', '23351.png', '00784.png', '11206.png', '10779.png', '10263.png', '03392.png', '15320.png', '16401.png', '18669.png', '24313.png', '22753.png', '20368.png', '18995.png', '05362.png', '20597.png', '00976.png', '24669.png', '16658.png', '06202.png', '20258.png', '03096.png', '00094.png', '17600.png', '03408.png', '21344.png', '11889.png', '11184.png', '18438.png', '06410.png', '24686.png', '04892.png', '02300.png', '01308.png', '21271.png', '09914.png', '05607.png', '21473.png', '24532.png', '10380.png', '01889.png', '23035.png', '03405.png', '12537.png', '23046.png', '05346.png', '09443.png', '11807.png', '21614.png', '15582.png', '11281.png', '22398.png', '01384.png', '05436.png', '11571.png', '06283.png', '17218.png', '00827.png', '18941.png', '05086.png', '24798.png', '15514.png', '24737.png', '15498.png', '07170.png', '05590.png', '07940.png', '01101.png', '05919.png', '02219.png', '14165.png', '24879.png', '13275.png', '00655.png', '06828.png', '17654.png', '02933.png', '12403.png', '02183.png', '14507.png', '05947.png', '24035.png', '02041.png', '02579.png', '16093.png', '10010.png', '24807.png', '17138.png', '15239.png', '02786.png', '13783.png', '11238.png', '13864.png', '07125.png', '10935.png', '20044.png', '13897.png', '21212.png', '20497.png', '15008.png', '00704.png', '11828.png', '19277.png', '18085.png', '16857.png', '15454.png', '12366.png', '16276.png', '02450.png', '04607.png', '09055.png', '09721.png', '06004.png', '04737.png', '16775.png', '02830.png', '18286.png', '02641.png', '04976.png', '24689.png', '19160.png', '19494.png', '20008.png', '23417.png', '00072.png', '16009.png', '21004.png', '23494.png', '21231.png', '08143.png', '18648.png', '07194.png', '23878.png', '14916.png', '15588.png', '20651.png', '18379.png', '11190.png', '22427.png', '04302.png', '00803.png', '05614.png', '13229.png', '09523.png', '02203.png', '10518.png', '10679.png', '00145.png', '08328.png', '20634.png', '01495.png', '15575.png', '22563.png', '19497.png', '23051.png', '03723.png', '15862.png', '03617.png', '11572.png', '09596.png', '12014.png', '06344.png', '14612.png', '10861.png', '14665.png', '22609.png', '24933.png', '06325.png', '15497.png', '17228.png', '09324.png', '11275.png', '10229.png', '15924.png', '12461.png', '14030.png', '22417.png', '04800.png', '13754.png', '13013.png', '11902.png', '04343.png', '08208.png', '01444.png', '15706.png', '13717.png', '14416.png', '15889.png', '09550.png', '05053.png', '00651.png', '15182.png', '17866.png', '14773.png', '08914.png', '10271.png', '01853.png', '10505.png', '11460.png', '00567.png', '02405.png', '17167.png', '06126.png', '11637.png', '04971.png', '13249.png', '04292.png', '14080.png', '16770.png', '16913.png', '10659.png', '10141.png', '19631.png', '19758.png', '24672.png', '02564.png', '06184.png', '12895.png', '24718.png', '10470.png', '24342.png', '19548.png', '23281.png', '08245.png', '04004.png', '19253.png', '00979.png', '12477.png', '05813.png', '05846.png', '01459.png', '06316.png', '02603.png', '05282.png', '12079.png', '01340.png', '11651.png', '15750.png', '04299.png', '03409.png', '14929.png', '11690.png', '13594.png', '06406.png', '09279.png', '22757.png', '24363.png', '22569.png', '00901.png', '12046.png', '14353.png', '16632.png', '10664.png', '06272.png', '23710.png', '08390.png', '18697.png', '05341.png', '21477.png', '01950.png', '04766.png', '10018.png', '13061.png', '18054.png', '19933.png', '13567.png', '11180.png', '04695.png', '13920.png', '24912.png', '20714.png', '14851.png', '10996.png', '24229.png', '19920.png', '06442.png', '03312.png', '23637.png', '12143.png', '21363.png', '07875.png', '12627.png', '02997.png', '01789.png', '13205.png', '05289.png', '01841.png', '09715.png', '16694.png', '18443.png', '08948.png', '09644.png', '00516.png', '19559.png', '01504.png', '00912.png', '10750.png', '24812.png', '16798.png', '10961.png', '17625.png', '04040.png', '20936.png', '04915.png', '21685.png', '11008.png', '16647.png', '23059.png', '06542.png', '19463.png', '23521.png', '07262.png', '07167.png', '02470.png', '03627.png', '18298.png', '01717.png', '23937.png', '13607.png', '12271.png', '06974.png', '12933.png', '10606.png', '05219.png', '20511.png', '16555.png', '14099.png', '12493.png', '01659.png', '05277.png', '03818.png', '21771.png', '09760.png', '09880.png', '15071.png', '14055.png', '07268.png', '01590.png', '10306.png', '23406.png', '13484.png', '03381.png', '15790.png', '10933.png', '16812.png', '04764.png', '18153.png', '16715.png', '15965.png', '18143.png', '24084.png', '11617.png', '14639.png', '00162.png', '23274.png', '01153.png', '12781.png', '03620.png', '03819.png', '14202.png', '21888.png', '08130.png', '05782.png', '15306.png', '22340.png', '00204.png', '22451.png', '02817.png', '02080.png', '17105.png', '13302.png', '13775.png', '04672.png', '24916.png', '05837.png', '14539.png', '21359.png', '05169.png', '14807.png', '13096.png', '15993.png', '23750.png', '03195.png', '18761.png', '07668.png', '14959.png', '22214.png', '07889.png', '18739.png', '04874.png', '20614.png', '20471.png', '20225.png', '16750.png', '02000.png', '15373.png', '06151.png', '13551.png', '24137.png', '14427.png', '01722.png', '00217.png', '22094.png', '01623.png', '10104.png', '20022.png', '01205.png', '20198.png', '04184.png', '09172.png', '23771.png', '07747.png', '12686.png', '05466.png', '20566.png', '05175.png', '20977.png', '21480.png', '24609.png', '19803.png', '01288.png', '21331.png', '10682.png', '20179.png', '19181.png', '17951.png', '10898.png', '09236.png', '19268.png', '13051.png', '07903.png', '12375.png', '13263.png', '07222.png', '19265.png', '02191.png', '20558.png', '07260.png', '08598.png', '06029.png', '20522.png', '15612.png', '18176.png', '01745.png', '05651.png', '11347.png', '10440.png', '15243.png', '09431.png', '04768.png', '12292.png', '14065.png', '17451.png', '11439.png', '10281.png', '24001.png', '20480.png', '15308.png', '04264.png', '12428.png', '06022.png', '04761.png', '14060.png', '21961.png', '14646.png', '10708.png', '06097.png', '06953.png', '10517.png', '15931.png', '07759.png', '16021.png', '14642.png', '11611.png', '09562.png', '17363.png', '00771.png', '09863.png', '17562.png', '00848.png', '20329.png', '19610.png', '01885.png', '13360.png', '21946.png', '15321.png', '18519.png', '05055.png', '21369.png', '23809.png', '15168.png', '15741.png', '11923.png', '19556.png', '08282.png', '03635.png', '09108.png', '24143.png', '13145.png', '24136.png', '01830.png', '20599.png', '13799.png', '17775.png', '08928.png', '24783.png', '03777.png', '11520.png', '14655.png', '05309.png', '17818.png', '19684.png', '11270.png', '02235.png', '17924.png', '03655.png', '17896.png', '20458.png', '16373.png', '06275.png', '01962.png', '05059.png', '03553.png', '05294.png', '21600.png', '01182.png', '07770.png', '17977.png', '10759.png', '17514.png', '10428.png', '22458.png', '21338.png', '17833.png', '19702.png', '13235.png', '20421.png', '22740.png', '06279.png', '22593.png', '01401.png', '06135.png', '13545.png', '23241.png', '19036.png', '08528.png', '24873.png', '17047.png', '03600.png', '22911.png', '01448.png', '13029.png', '06568.png', '07395.png', '08632.png', '03069.png', '10743.png', '24265.png', '15812.png', '21474.png', '20691.png', '08599.png', '15342.png', '22566.png', '13344.png', '13311.png', '05715.png', '16318.png', '02424.png', '03431.png', '23340.png', '02915.png', '07445.png', '12193.png', '01723.png', '21163.png', '05127.png', '08457.png', '19040.png', '10911.png', '05371.png', '05041.png', '23863.png', '15121.png', '21394.png', '15860.png', '22841.png', '10185.png', '23397.png', '19064.png', '04888.png', '17315.png', '14340.png', '16197.png', '13396.png', '07100.png', '20860.png', '24803.png', '01956.png', '10124.png', '08927.png', '04937.png', '20715.png', '13867.png', '08845.png', '24881.png', '05814.png', '05568.png', '22788.png', '19990.png', '01833.png', '23528.png', '00316.png', '23053.png', '11322.png', '06841.png', '23209.png', '12439.png', '06990.png', '06573.png', '13173.png', '09276.png', '18447.png', '00992.png', '11771.png', '07112.png', '08617.png', '19302.png', '14560.png', '02921.png', '19570.png', '14496.png', '22349.png', '03824.png', '03995.png', '21202.png', '10398.png', '05725.png', '18615.png', '21188.png', '01728.png', '00349.png', '08239.png', '03829.png', '03750.png', '11458.png', '21257.png', '01582.png', '05144.png', '05721.png', '23936.png', '16296.png', '21489.png', '09573.png', '20873.png', '16778.png', '24221.png', '00244.png', '08625.png', '15376.png', '08900.png', '12360.png', '01985.png', '04142.png', '07782.png', '24738.png', '02338.png', '00689.png', '06966.png', '18802.png', '23025.png', '13895.png', '08444.png', '17022.png', '04229.png', '05844.png', '20284.png', '22677.png', '01817.png', '07129.png', '23816.png', '04341.png', '15672.png', '23448.png', '19167.png', '19849.png', '19909.png', '21511.png', '23012.png', '05997.png', '18303.png', '05525.png', '00701.png', '12084.png', '22154.png', '13455.png', '17939.png', '03382.png', '05401.png', '04724.png', '09157.png', '03614.png', '01014.png', '05811.png', '03100.png', '22286.png', '22139.png', '16061.png', '24629.png', '04110.png', '15907.png', '03307.png', '04477.png', '19853.png', '06446.png', '09127.png', '19784.png', '14831.png', '15094.png', '15333.png', '09625.png', '06794.png', '22795.png', '16324.png', '12759.png', '17574.png', '05746.png', '12979.png', '03477.png', '22085.png', '05645.png', '09241.png', '13357.png', '17230.png', '24448.png', '09188.png', '02849.png', '10430.png', '22658.png', '07487.png', '20881.png', '21328.png', '14169.png', '21668.png', '09307.png', '12278.png', '23154.png', '05299.png', '20308.png', '11396.png', '02193.png', '24599.png', '01431.png', '06132.png', '08421.png', '22825.png', '18814.png', '18097.png', '23658.png', '05819.png', '18297.png', '16038.png', '10340.png', '09876.png', '12343.png', '23160.png', '17532.png', '15655.png', '07970.png', '22145.png', '01025.png', '15372.png', '12479.png', '22717.png', '11920.png', '02551.png', '22485.png', '08063.png', '14092.png', '08166.png', '23648.png', '03300.png', '12936.png', '08806.png', '01100.png', '23888.png', '22842.png', '02602.png', '07846.png', '15988.png', '21339.png', '12285.png', '13741.png', '01421.png', '09315.png', '04731.png', '13105.png', '20300.png', '24539.png', '23064.png', '03338.png', '14499.png', '00029.png', '22891.png', '22857.png', '06866.png', '20390.png', '23995.png', '00733.png', '16448.png', '20261.png', '12642.png', '13178.png', '22024.png', '10781.png', '08064.png', '01910.png', '02628.png', '12107.png', '00632.png', '18869.png', '20993.png', '15140.png', '01136.png', '18968.png', '23356.png', '18713.png', '04234.png', '17231.png', '22858.png', '22678.png', '00820.png', '12196.png', '02473.png', '20000.png', '16062.png', '06788.png', '06918.png', '06492.png', '19130.png', '07561.png', '19436.png', '16783.png', '03130.png', '07009.png', '18603.png', '02733.png', '17906.png', '10365.png', '06926.png', '17910.png', '18306.png', '18586.png', '08308.png', '23929.png', '12734.png', '20316.png', '12604.png', '09255.png', '15583.png', '17993.png', '24541.png', '17338.png', '16744.png', '22936.png', '14828.png', '18139.png', '22530.png', '19995.png', '02134.png', '17614.png', '03047.png', '07692.png', '13007.png', '08583.png', '10739.png', '07161.png', '14341.png', '21080.png', '08989.png', '20768.png', '08960.png', '23318.png', '04055.png', '21566.png', '11036.png', '02772.png', '04569.png', '13285.png', '19652.png', '22662.png', '19115.png', '07325.png', '23005.png', '12006.png', '19805.png', '00698.png', '10291.png', '15394.png', '06842.png', '18492.png', '12576.png', '09694.png', '18691.png', '10956.png', '03716.png', '09350.png', '02715.png', '14867.png', '02408.png', '16440.png', '04751.png', '00659.png', '24677.png', '17688.png', '00446.png', '01294.png', '13971.png', '17972.png', '16162.png', '11579.png', '21928.png', '20443.png', '07474.png', '03754.png', '18932.png', '06581.png', '23267.png', '12423.png', '23510.png', '02648.png', '21524.png', '00442.png', '10625.png', '11796.png', '02669.png', '03532.png', '13442.png', '05045.png', '21055.png', '22009.png', '07709.png', '15867.png', '07179.png', '10723.png', '23541.png', '01075.png', '04392.png', '16506.png', '08349.png', '16041.png', '02190.png', '20505.png', '04282.png', '20565.png', '16137.png', '09401.png', '23305.png', '08818.png', '09284.png', '18295.png', '13390.png', '03121.png', '13901.png', '05571.png', '08475.png', '22037.png', '02260.png', '04675.png', '12244.png', '05273.png', '17094.png', '06121.png', '07500.png', '05998.png', '14491.png', '05224.png', '01299.png', '16002.png', '06226.png', '24858.png', '15667.png', '17465.png', '24062.png', '24864.png', '20091.png', '00296.png', '00838.png', '12543.png', '02546.png', '10652.png', '02262.png', '23805.png', '01502.png', '17538.png', '02043.png', '07055.png', '20243.png', '03284.png', '08980.png', '16968.png', '09432.png', '05697.png', '03796.png', '09477.png', '10802.png', '07076.png', '17844.png', '15680.png', '24239.png', '01057.png', '11904.png', '13733.png', '06859.png', '12134.png', '08544.png', '07592.png', '20380.png', '19683.png', '02853.png', '04372.png', '11099.png', '06524.png', '18137.png', '16134.png', '07917.png', '10979.png', '09565.png', '18020.png', '11837.png', '19059.png', '12940.png', '18198.png', '05030.png', '22369.png']    
    epochs_since_start = 0
    if True:
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None
        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN, a = a)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)
    print('gta size:',len(trainloader))
    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()
    model.cuda()
    model.train()
    #prototype_dist_init(cfg, trainloader, model)
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0
    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)
    
    """
    if True:
        model.eval()
        if dataset == 'cityscapes':
            mIoU, eval_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024))

        model.train()
        print("mIoU: ",mIoU, eval_loss)
    """
    
    accumulated_loss_l = []
    accumulated_loss_u = []
    accumulated_loss_feat = []
    accumulated_loss_out = []   
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    
    print(epochs_since_start)
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        loss_feat_value = 0
        loss_out_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            if epochs_since_start >= 2:
                list_name = []
            if epochs_since_start == 1:
                data_loader = get_loader('gta')
                data_path = get_data_path('gta')
                if random_crop:
                    data_aug = Compose([RandomCrop_gta(input_size)])
                else:
                    data_aug = None        
                train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN, a = None)
                trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                print('gta size:',len(trainloader))
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        #if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        #else:
        
        weak_parameters={"flip": 0}


        images, labels, _, names = batch
        images = images.cuda()
        labels = labels.cuda().long()
        if epochs_since_start >= 2:
            for name in names:
                list_name.append(name)

        #images, labels = weakTransform(weak_parameters, data = images, target = labels)

        src_pred, src_feat= model(images)
        pred = interp(src_pred)
        L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data
        #L_l = torch.Tensor([0.0]).cuda()

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            #inputs_u_w = inputs_u_w.clone()
            logits_u_w = interp(ema_model(inputs_u_w)[0])
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            if mix_mask == "class":
                for image_i in range(batch_size):
                    classes = torch.unique(labels[image_i])
                    #classes=classes[classes!=ignore_label]
                    nclasses = classes.shape[0]
                    #if nclasses > 0:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()

                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))

            strong_parameters = {"Mix": MixMask0}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            inputs_u_s1, _ = strongTransform(strong_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
            inputs_u_s = torch.cat((inputs_u_s0,inputs_u_s1))
            logits_u_s_tgt, tgt_feat = model(inputs_u_s)
            logits_u_s = interp(logits_u_s_tgt)

            strong_parameters["Mix"] = MixMask0
            _, targets_u0 = strongTransform(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_u_w[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, targets_u1 = strongTransform(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_u_w[1].unsqueeze(0))))
            targets_u = torch.cat((targets_u0,targets_u1)).long()
            
            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float().cuda()
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).cuda()

            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            strong_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
            pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s, targets_u, pixelWiseWeight)

            loss = L_l + L_u

        else:
            loss = L_l
        
        # source mask: downsample the ground-truth label
        src_out_ema, src_feat_ema = ema_model(images)
        tgt_out_ema, tgt_feat_ema = ema_model(inputs_u_s)
        B, A, Hs, Ws = src_feat.size()
        src_mask = F.interpolate(labels.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        assert not src_mask.requires_grad
        pseudo_weight = F.interpolate(pixelWiseWeight.unsqueeze(1),
                                         size=(65,65), mode='bilinear',
                                         align_corners=True).squeeze(1)
        
        _, _, Ht, Wt = tgt_feat.size()
        tgt_out_maxvalue, tgt_mask_st = torch.max(tgt_feat_ema, dim=1)
        tgt_mask = F.interpolate(targets_u.unsqueeze(1).float(), size=(65,65), mode='nearest').squeeze(1).long()
        tgt_mask_upt = copy.deepcopy(tgt_mask)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_mask_upt[(((tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask_st == i)).int() + (pseudo_weight != 1.0).int()) == 2] = 255

        tgt_mask = tgt_mask.contiguous().view(B * Hs * Ws, )
        pseudo_weight = pseudo_weight.contiguous().view(B * Hs * Ws, )
        tgt_mask_upt = tgt_mask_upt.contiguous().view(B * Hs * Ws, )
        src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
        src_feat_ema = src_feat_ema.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat_ema = tgt_feat_ema.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        # update feature-level statistics
        feat_estimator.update(features=tgt_feat_ema.detach(), labels=tgt_mask_upt)
        feat_estimator.update(features=src_feat_ema.detach(), labels=src_mask)

        # contrastive loss on both domains
        
        loss_feat = pcl_criterion_src(Proto=feat_estimator.Proto.detach(),
                                  feat=src_feat,
                                  labels=src_mask) \
                    + pcl_criterion_tgt(Proto=feat_estimator.Proto.detach(),
                                  feat=tgt_feat,
                                  labels=tgt_mask, pixelWiseWeight=pseudo_weight)
        #meters.update(loss_feat=loss_feat.item())

        if cfg.SOLVER.MULTI_LEVEL:
            src_out = src_pred.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out = logits_u_s_tgt.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
            src_out_ema = src_out_ema.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out_ema = tgt_out_ema.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
            # update output-level statistics
            out_estimator.update(features=tgt_out_ema.detach(), labels=tgt_mask_upt)
            out_estimator.update(features=src_out_ema.detach(), labels=src_mask)

            # the proposed contrastive loss on prediction map
            loss_out = pcl_criterion_src(Proto=out_estimator.Proto.detach(),
                                     feat=src_out,
                                     labels=src_mask) \
                       + pcl_criterion_tgt(Proto=out_estimator.Proto.detach(),
                                       feat=tgt_out,
                                       labels=tgt_mask, pixelWiseWeight=pseudo_weight)
            #meters.update(loss_out=loss_out.item())

            loss = loss + cfg.SOLVER.LAMBDA_FEAT * loss_feat + cfg.SOLVER.LAMBDA_OUT * loss_out
        else:
            loss = loss + cfg.SOLVER.LAMBDA_FEAT * loss_feat

        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            if train_unlabeled:
                loss_u_value += L_u.item()
            loss_feat_value += loss_feat.item()
            loss_out_value += loss_out.item()
        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)

        #print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))

        if i_iter % save_checkpoint_every == 1486 and i_iter!=0:
            _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            feat_estimator.save(name='prototype_feat_dist.pth')
            out_estimator.save(name='prototype_out_dist.pth')
            print('save_prototype')

        if i_iter == 169517:
            print(list_name)
        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

        accumulated_loss_l.append(loss_l_value)
        accumulated_loss_feat.append(loss_feat_value)
        accumulated_loss_out.append(loss_out_value)
        if train_unlabeled:
            accumulated_loss_u.append(loss_u_value)
            
        if i_iter % log_per_iter == 0 and i_iter != 0:
                #tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
            print('Training/contrastive_feat_loss', np.mean(accumulated_loss_feat), 'Training/contrastive_out_loss', np.mean(accumulated_loss_out), i_iter)
            accumulated_loss_feat = []
            accumulated_loss_out = []
            if train_unlabeled:
                #tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                print('Training/Supervised loss', np.mean(accumulated_loss_l), 'Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                accumulated_loss_u = []
                accumulated_loss_l = []
            
        if save_unlabeled_images and train_unlabeled and (i_iter == 5650600):
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s[0].cpu(),i_iter,'input_s1',palette.CityScpates_palette)
            save_image(inputs_u_s[1].cpu(),i_iter,'input_s2',palette.CityScpates_palette)
            save_image(inputs_u_w[0].cpu(),i_iter,'input_w1',palette.CityScpates_palette)
            save_image(inputs_u_w[1].cpu(),i_iter,'input_w2',palette.CityScpates_palette)
            save_image(images[0].cpu(),i_iter,'input1',palette.CityScpates_palette)
            save_image(images[1].cpu(),i_iter,'input2',palette.CityScpates_palette)

            _, pred_u_s = torch.max(logits_u_w, dim=1)
            #save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
            #save_image(pred_u_s[1].cpu(),i_iter,'pred2',palette.CityScpates_palette)

    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    if dataset == 'cityscapes':
        mIoU, val_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024), save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)


    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()
    

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] 

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    #if args.save_images:
    print('Saving unlabeled images')
    save_unlabeled_images = True
    #else:
    #save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]    
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()


    main()
