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
    a = ['00778.png', '13101.png', '24360.png', '02841.png', '13224.png', '17022.png', '22567.png', '09364.png', '21730.png', '22909.png', '02683.png', '18704.png', '24426.png', '12743.png', '07411.png', '16920.png', '06774.png', '01363.png', '18070.png', '07531.png', '12133.png', '13826.png', '17024.png', '17023.png', '05987.png', '06962.png', '02995.png', '16739.png', '09873.png', '23735.png', '12376.png', '00726.png', '08243.png', '07742.png', '02645.png', '11466.png', '11530.png', '20099.png', '20614.png', '15944.png', '16304.png', '22568.png', '08784.png', '06787.png', '11700.png', '18736.png', '06030.png', '13570.png', '02157.png', '11967.png', '20239.png', '22625.png', '23659.png', '07885.png', '13108.png', '16581.png', '10850.png', '02199.png', '10997.png', '09029.png', '19749.png', '22913.png', '01355.png', '08202.png', '15395.png', '24171.png', '15701.png', '21990.png', '05118.png', '11757.png', 
'10783.png', '15892.png', '01722.png', '17871.png', '24619.png', '18410.png', '19719.png', '09850.png', '01880.png', '13655.png', '17829.png', '20535.png', '19017.png', '00009.png', '10487.png', '01407.png', '23828.png', '21844.png', '11788.png', '04504.png', '02729.png', '01025.png', '00198.png', '03207.png', '20676.png', '15358.png', '17527.png', '17581.png', '13845.png', '24248.png', '09800.png', '12246.png', '12427.png', '22019.png', '17744.png', '04062.png', '05724.png', '00247.png', '21271.png', '08222.png', '12620.png', '14517.png', '17094.png', '07679.png', '06184.png', '17341.png', '16178.png', '21067.png', '02887.png', '22721.png', '03072.png', '08970.png', '19789.png', '18781.png', '23615.png', '12784.png', '17970.png', '09606.png', '18651.png', '12218.png', '16668.png', '11317.png', '11195.png', '12575.png', '19515.png', '13243.png', '16064.png', '09243.png', '14567.png', '04734.png', '00389.png', '15240.png', '09148.png', '01617.png', '12721.png', '10534.png', '04230.png', '20311.png', '02688.png', '05385.png', '15946.png', '20484.png', '07461.png', '08665.png', '04512.png', '10293.png', '04199.png', '18792.png', '07576.png', '18659.png', '17699.png', '16145.png', '23630.png', '01043.png', '07621.png', '11301.png', '19716.png', '09898.png', '18230.png', '22196.png', '23733.png', '11403.png', '08771.png', '01350.png', '16007.png', '02620.png', '16570.png', '17290.png', '13384.png', '07859.png', '16049.png', '21287.png', '21080.png', '23907.png', '00230.png', '18503.png', '09307.png', '15859.png', '10883.png', '20885.png', '09267.png', '01486.png', '05327.png', '14102.png', '04802.png', '19278.png', '08625.png', '24602.png', '03202.png', '21422.png', '12825.png', '00520.png', '21825.png', '24561.png', '10239.png', '15237.png', '15025.png', '21852.png', '07993.png', '14336.png', '20729.png', '13974.png', '17491.png', '21004.png', '20720.png', '21261.png', '09396.png', '11869.png', '16994.png', '11009.png', '23342.png', '00717.png', '18845.png', '04646.png', '08796.png', '06633.png', '17523.png', '09390.png', '07474.png', '18712.png', '23407.png', '11187.png', '09730.png', '18695.png', '01028.png', '16448.png', '06027.png', '07747.png', '13300.png', '14790.png', '16676.png', '10778.png', '04052.png', '00255.png', '08209.png', '10587.png', '00026.png', '21769.png', '12742.png', '23379.png', '00333.png', '02783.png', '23067.png', '18548.png', '14691.png', '15375.png', '09343.png', '10958.png', '13810.png', '17199.png', '14847.png', '21647.png', '03760.png', '07863.png', '10862.png', '03206.png', '01728.png', '08907.png', '18028.png', '07628.png', '13395.png', '06105.png', '08536.png', '14257.png', '10011.png', '12083.png', '06692.png', '13942.png', '12423.png', '20957.png', '18110.png', '22734.png', '00718.png', '00928.png', '19221.png', '23018.png', '09462.png', '00021.png', '00441.png', '12369.png', '11988.png', '15554.png', '13246.png', '19585.png', '06502.png', '18895.png', '07573.png', '18199.png', '05564.png', '06338.png', '16432.png', '14687.png', '02379.png', '16974.png', '02001.png', '16455.png', '12725.png', '10970.png', '00515.png', '19456.png', '16534.png', '19786.png', '04828.png', '18886.png', '04447.png', '01651.png', '06132.png', '13353.png', '07709.png', '09815.png', '14934.png', '03498.png', '23408.png', '11586.png', '02174.png', '06812.png', '11775.png', '05491.png', '06645.png', '24812.png', '15868.png', '03835.png', '06580.png', '10885.png', '05009.png', '20340.png', '02149.png', '08780.png', '24185.png', '20382.png', '17296.png', '20591.png', '21904.png', '16963.png', '06868.png', '12032.png', '18131.png', '00919.png', '03485.png', '15684.png', '16057.png', '15543.png', '10386.png', '22365.png', '15329.png', '06726.png', '01870.png', '10384.png', '13233.png', '24043.png', '07433.png', '04466.png', '18122.png', '17354.png', '24889.png', '16472.png', '15573.png', '02855.png', '10751.png', '18584.png', '04046.png', '06009.png', '11272.png', '15318.png', '18542.png', '14181.png', '02202.png', '07975.png', '02694.png', '10521.png', '15672.png', '20179.png', '11976.png', '17816.png', '09132.png', '08512.png', '00604.png', '11499.png', '21711.png', '10324.png', '04147.png', '19347.png', '08569.png', '14201.png', '12066.png', '09185.png', '17451.png', '00932.png', '19216.png', '14907.png', '00325.png', '02565.png', '00036.png', '06243.png', '08214.png', '16514.png', '17046.png', '11814.png', '09317.png', '13540.png', '00467.png', '14468.png', '24780.png', '01071.png', '17740.png', '03978.png', '11142.png', '22308.png', '03490.png', '21836.png', '01817.png', '20929.png', '19854.png', '06357.png', '12159.png', '21929.png', '22064.png', '02988.png', '05299.png', '22380.png', '01628.png', '15702.png', '06065.png', '04498.png', '03798.png', '01720.png', '22358.png', '01816.png', '03621.png', '02004.png', '00986.png', '15879.png', '03058.png', '16713.png', '13164.png', '08248.png', '20516.png', '08410.png', '00320.png', '12806.png', '01707.png', '10833.png', '23717.png', '10857.png', '02947.png', '02233.png', '15695.png', '06891.png', '01781.png', '02738.png', '05035.png', '07109.png', '02643.png', '00607.png', '06282.png', '02154.png', '07295.png', '10034.png', '03327.png', '09768.png', '15462.png', '14771.png', '16495.png', '07512.png', '20221.png', '09590.png', '04309.png', '00901.png', '12411.png', '22907.png', '24314.png', '01836.png', '03793.png', '12069.png', '16625.png', '21065.png', '21323.png', '01935.png', '19468.png', '10744.png', '23688.png', '10108.png', '04612.png', '05931.png', '06321.png', '20031.png', '22179.png', '05631.png', '02518.png', '22640.png', '16193.png', '07766.png', '02943.png', '02666.png', '00746.png', '04395.png', '04905.png', '20152.png', '20858.png', '24736.png', '00297.png', '19935.png', '06119.png', '06197.png', '14129.png', '17945.png', '17219.png', '04628.png', '04174.png', '11482.png', '12850.png', '14814.png', '16200.png', '17848.png', '09021.png', '13684.png', '11274.png', '13911.png', '04701.png', '16140.png', '13556.png', '22728.png', '22467.png', '16417.png', '02429.png', '16957.png', '10085.png', '19316.png', '03853.png', '06076.png', '22246.png', '03271.png', '16893.png', '00995.png', '09772.png', '21594.png', '19677.png', '01583.png', '10495.png', '17034.png', '04951.png', '02403.png', '17113.png', '19319.png', '15901.png', '11743.png', '13029.png', '03656.png', '24264.png', '13438.png', '15766.png', '12454.png', '07409.png', '12738.png', '16325.png', '03419.png', '00489.png', '05534.png', '07595.png', '10200.png', '21407.png', '00084.png', '03985.png', '05930.png', '20141.png', '03923.png', '16749.png', '00004.png', '20765.png', '12775.png', '07793.png', '19426.png', '07852.png', '19830.png', '03306.png', '12058.png', '07690.png', '21722.png', '17284.png', '09999.png', '24597.png', '23793.png', '22948.png', '16631.png', '16230.png', '22500.png', '09864.png', '03518.png', '18512.png', '08394.png', '21513.png', '18784.png', '13723.png', '09254.png', '17446.png', '03187.png', '02488.png', '00091.png', '01884.png', '20998.png', '06317.png', '06940.png', '10681.png', '12947.png', '01945.png', '16723.png', '02377.png', '12840.png', '12388.png', '14474.png', '19073.png', '08727.png', '13376.png', '18102.png', '12905.png', '05545.png', '04280.png', '12357.png', '09944.png', '11343.png', '12115.png', '14620.png', '15469.png', '17189.png', '02871.png', '23833.png', '08551.png', '22293.png', '03221.png', '05072.png', '15947.png', '01045.png', '15208.png', '10302.png', '09780.png', '10052.png', '00910.png', '13650.png', '20034.png', '09056.png', '05388.png', '11649.png', '06863.png', '19133.png', '00972.png', '10485.png', '11912.png', '08504.png', '12961.png', '04674.png', '23511.png', '05751.png', '18135.png', '16812.png', '07363.png', '18835.png', '20266.png', '18159.png', '08729.png', '06331.png', '08852.png', '04407.png', '23423.png', '19645.png', '17289.png', '05100.png', '13792.png', '16328.png', '16741.png', '15099.png', '00519.png', '17650.png', '03264.png', '11644.png', '13491.png', '11728.png', '21692.png', '00360.png', '06656.png', '09091.png', '14174.png', '01537.png', '03702.png', '05222.png', '09859.png', '20411.png', '07871.png', '02066.png', '01224.png', '23277.png', '13972.png', '22311.png', '13805.png', '05304.png', '12842.png', '01834.png', '19457.png', '09890.png', '11136.png', '21041.png', '04133.png', '06924.png', '06751.png', '11105.png', '05162.png', '16560.png', '00311.png', '05619.png', '00568.png', '22661.png', '00625.png', '18283.png', '15539.png', '08663.png', '15673.png', '08870.png', '15727.png', '04501.png', '08773.png', '22287.png', '03190.png', '16620.png', '10960.png', '16626.png', '14054.png', '02991.png', '21466.png', '08166.png', '11872.png', '01293.png', '00621.png', '12718.png', '18759.png', '13637.png', '21532.png', '13721.png', '24517.png', '11488.png', '16610.png', '23588.png', '15091.png', '21451.png', '10497.png', '04526.png', '15767.png', '00656.png', '14794.png', '23145.png', '16068.png', '23266.png', '05185.png', '01871.png', '09931.png', '15842.png', '04030.png', '05045.png', '11897.png', '15829.png', '20749.png', '10317.png', '18557.png', '09406.png', '18194.png', '14014.png', '03689.png', '17923.png', '18391.png', '15425.png', '03968.png', '11998.png', '03947.png', '09040.png', '15718.png', '14217.png', '23980.png', '23865.png', '21975.png', '13183.png', '05551.png', '05346.png', '07094.png', '19024.png', '13559.png', '18881.png', '13828.png', '19284.png', '16399.png', '10852.png', '16351.png', '14060.png', '04402.png', '05196.png', '24405.png', '23160.png', '10613.png', '18849.png', '09597.png', '13744.png', '08778.png', '22830.png', '10516.png', '19752.png', '04531.png', '17879.png', '05548.png', '02857.png', '03162.png', '09112.png', '13584.png', '15475.png', '24859.png', '15248.png', '22831.png', '11464.png', '07660.png', '23878.png', '21309.png', '05160.png', '13753.png', '19897.png', '19304.png', '10004.png', '02549.png', '07692.png', '03036.png', '10690.png', '22674.png', '02573.png', '03377.png', '00071.png', '00711.png', '22418.png', '15092.png', '02940.png', '09383.png', '03566.png', '05898.png', '03392.png', '03908.png', '08913.png', '02064.png', '08891.png', '18861.png', '04136.png', '22692.png', '15314.png', '22329.png', '17292.png', '13138.png', '09822.png', '14137.png', '14590.png', '22150.png', '06721.png', '23488.png', '04002.png', '02720.png', '04235.png', '06801.png', '06306.png', '17188.png', '07103.png', '02458.png', '11881.png', '00423.png', '13455.png', '14000.png', '08060.png', '15301.png', '21814.png', '09116.png', '23536.png', '16840.png', '12726.png', '09228.png', '13192.png', '11639.png', '17415.png', '02865.png', '23062.png', '16879.png', '09548.png', '18324.png', '03869.png', '24843.png', '22143.png', '13635.png', '05350.png', '19453.png', '13305.png', '15881.png', '01612.png', '22785.png', '24179.png', '04130.png', '12005.png', '15874.png', '07371.png', '15051.png', '07634.png', '00309.png', '10579.png', '22580.png', '09141.png', '00473.png', '10756.png', '05889.png', '19290.png', '21139.png', '00201.png', '03548.png', '06589.png', '14145.png', '11684.png', '00216.png', '11289.png', '13462.png', '10437.png', '23055.png', '05453.png', '20268.png', '12110.png', '16125.png', '12949.png', '06226.png', '00558.png', '07009.png', '10328.png', '10944.png', '23493.png', '24712.png', '22968.png', '24374.png', '01973.png', '02827.png', '00759.png', '17309.png', '07598.png', '09299.png', '07636.png', '04348.png', '24181.png', '01078.png', '03434.png', '21615.png', '00212.png', '11148.png', '19584.png', '03989.png', '10446.png', '06240.png', '01822.png', '03382.png', '03307.png', '20785.png', '20011.png', '07864.png', '03721.png', '11022.png', '17464.png', '04253.png', '15498.png', '18287.png', '07264.png', '19171.png', '09952.png', '20876.png', '00403.png', '07485.png', '01451.png', '11158.png', '07204.png', '09473.png', '08184.png', '01227.png', '08302.png', '02802.png', '19939.png', '24601.png', '22783.png', '22531.png', '16708.png', '01526.png', '11620.png', '13149.png', '17460.png', '04975.png', '01018.png', '04652.png', '22226.png', '23670.png', '18295.png', '20467.png', '17389.png', '07977.png', '08714.png', '17956.png', '13413.png', '09719.png', '02863.png', '17727.png', '01160.png', '12086.png', '14698.png', '11810.png', '10682.png', '06307.png', '19593.png', '00584.png', '16794.png', '12060.png', '16214.png', '00175.png', '05958.png', '21457.png', '18316.png', '23778.png', '18420.png', '20900.png', '19445.png', '09142.png', '11100.png', '00436.png', '15774.png', '13653.png', '06162.png', '07732.png', '05769.png', '05654.png', '21345.png', '22389.png', '08827.png', '17070.png', '16857.png', '05203.png', '03680.png', '13090.png', '12009.png', '06815.png', '14459.png', '20882.png', '07155.png', '24661.png', '24672.png', '19231.png', '21813.png', '14857.png', '23258.png', '22344.png', '01579.png', '23162.png', '21207.png', '17874.png', '07282.png', '07673.png', '20766.png', '09411.png', '19140.png', '13034.png', '10597.png', '06833.png', '08259.png', '07265.png', '04914.png', '19346.png', '07137.png', '05533.png', '12733.png', '09721.png', '09098.png', '11619.png', '23928.png', '04099.png', '23063.png', '04006.png', '24023.png', '05149.png', '01701.png', '18922.png', '09492.png', '11437.png', '05811.png', '10687.png', '10392.png', '11115.png', '19665.png', '08367.png', '07798.png', '00229.png', '05667.png', '23280.png', '07519.png', '22815.png', '16376.png', '22921.png', '18987.png', '17343.png', '01680.png', '17177.png', '21085.png', '00789.png', '02990.png', '15565.png', '00365.png', '20730.png', '10917.png', '18041.png', '22527.png', '20963.png', '08234.png', '05050.png', '10800.png', '18981.png', '20759.png', '06870.png', '18034.png', '24097.png', '24728.png', '00708.png', '21591.png', '07707.png', '07541.png', '10064.png', '19963.png', '18229.png', '24176.png', '18916.png', '23199.png', '24852.png', '02621.png', '02915.png', '04834.png', '06449.png', '06547.png', '12843.png', '18289.png', '24087.png', '01503.png', '12528.png', '12488.png', '02687.png', '09984.png', '21253.png', '20773.png', '06345.png', '09807.png', '18806.png', '02747.png', '02525.png', '16583.png', '22713.png', '12253.png', '24403.png', '17291.png', '11871.png', '06146.png', '04340.png', '01819.png', '13923.png', '06261.png', '08364.png', '04698.png', '12766.png', '07804.png', '10498.png', '06124.png', '19709.png', '01619.png', '12013.png', '09849.png', '22220.png', '23495.png', '16192.png', '21246.png', '21342.png', '04459.png', '01384.png', '10000.png', '04593.png', '11977.png', '17838.png', '19047.png', '07705.png', '01167.png', '24047.png', '19046.png', '19252.png', '20996.png', '21473.png', '13072.png', '09947.png', '09337.png', '07280.png', '13017.png', '04075.png', '10686.png', '09027.png', '15574.png', '19466.png', '06634.png', '07540.png', '23999.png', '13984.png', '23579.png', '13279.png', '01300.png', '08022.png', '18485.png', '22330.png', '02832.png', '20167.png', '02128.png', '21736.png', '05539.png', '06168.png', '21401.png', '08428.png', '07390.png', '12557.png', '00630.png', '24257.png', '01758.png', '23810.png', '00554.png', '22709.png', '14678.png', '22321.png', '04354.png', '19745.png', '13295.png', '13086.png', '07464.png', '01170.png', '03391.png', '17039.png', '16531.png', '01925.png', '08154.png', '19275.png', '21435.png', '01697.png', '20079.png', '03128.png', '24432.png', '16201.png', '02442.png', '13495.png', '09509.png', '10670.png', '24573.png', '22084.png', '23715.png', '14682.png', '04993.png', '20280.png', '07174.png', '22519.png', '09650.png', '01180.png', '04134.png', '03935.png', '03325.png', '19204.png', '15551.png', '20247.png', '04939.png', '04343.png', '10907.png', '24866.png', '19893.png', '09465.png', '13689.png', '04974.png', '21069.png', '07048.png', '19508.png', '20249.png', '23556.png', '02050.png', '05323.png', '16938.png', '21160.png', '08099.png', '04826.png', '11655.png', '05777.png', '00147.png', '21027.png', '21957.png', '04818.png', '03321.png', '00968.png', '12857.png', '22190.png', '03851.png', '23497.png', '22474.png', '13692.png', '03360.png', '13115.png', '07489.png', '18328.png', '08124.png', '09708.png', '03610.png', '12823.png', '18963.png', '02324.png', '06494.png', '12041.png', '07079.png', '16122.png', '10133.png', '08153.png', '23089.png', '09338.png', '00028.png', '24069.png', '22976.png', '02028.png', '04111.png', '19180.png', '23231.png', '15127.png', '00148.png', '16853.png', '16647.png', '12629.png', '20192.png', '02219.png', '21314.png', '06657.png', '00124.png', '19973.png', '00654.png', '16679.png', '12604.png', '11842.png', '12254.png', '15567.png', '15074.png', '02822.png', '15396.png', '13369.png', '12129.png', '00444.png', '01606.png', '06454.png', '16784.png', '03516.png', '04011.png', '04187.png', '20976.png', '23957.png', '06955.png', '20452.png', '01602.png', '24073.png', '23955.png', '08623.png', '08873.png', '10282.png', '16023.png', '01100.png', '00969.png', '19185.png', '11391.png', '11660.png', '01806.png', '03339.png', '23640.png', '15522.png', '13834.png', '00011.png', '10424.png', '21596.png', '06031.png', '19155.png', '23189.png', '01383.png', '09092.png', '11723.png', '05494.png', '03408.png', '03290.png', '19403.png', '00981.png', '10522.png', '10661.png', '22834.png', '16087.png', '11840.png', '24017.png', '03627.png', '13365.png', '06599.png', '19850.png', '04858.png', '17987.png', '12479.png', '23209.png', '18036.png', '01907.png', '22672.png', '17476.png', '10264.png', '04956.png', '09966.png', '21709.png', '18682.png', '14659.png', '16086.png', '05300.png', '09147.png', '23873.png', '09587.png', '24752.png', '20412.png', '04700.png', '04256.png', '04022.png', '11679.png', '16293.png', '04528.png', '04261.png', '18271.png', '13829.png', '18455.png', '03491.png', '10556.png', '13869.png', '12559.png', '03288.png', '19175.png', '03314.png', '02065.png', '18164.png', '19362.png', '19974.png', '06564.png', '15212.png', '10890.png', '21992.png', '14612.png', '06989.png', '00904.png', '03990.png', '13628.png', '20907.png', '00088.png', '11675.png', '12668.png', '06410.png', '06545.png', '20256.png', '07839.png', '24652.png', '10506.png', '12469.png', '05815.png', '17676.png', '12618.png', '04042.png', '12761.png', '19574.png', '00900.png', '07829.png', '06839.png', '15224.png', '12412.png', '05954.png', '01724.png', '24562.png', '01292.png', '23480.png', '23139.png', '23204.png', '14114.png', '05307.png', '04298.png', '23769.png', '03623.png', '24875.png', '19770.png', '03885.png', '17965.png', '04592.png', '21248.png', '07090.png', '15957.png', '07661.png', '19934.png', '10526.png', '01988.png', '01262.png', '11010.png', '04508.png', '06179.png', '06447.png', '05183.png', '19358.png', '08857.png', '08142.png', '07962.png', '18864.png', '23228.png', '02410.png', '04942.png', '03418.png', '21179.png', '08151.png', '03640.png', '11865.png', '10601.png', '11576.png', '02346.png', '06374.png', '21962.png', '04323.png', '04867.png', '13061.png', '06202.png', '13050.png', '00118.png', '17624.png', '20813.png', '17152.png', '16266.png', '14882.png', '07799.png', '24363.png', '10877.png', '11222.png', '07841.png', '02271.png', '11894.png', '23015.png', '11059.png', '11745.png', '16245.png', '08860.png', '18253.png', '17317.png', '24109.png', '13900.png', '14553.png', '18636.png', '11561.png', '15810.png', '12432.png', '00662.png', '07463.png', '08812.png', '21068.png', '20571.png', '14721.png', '11535.png', '20603.png', '07205.png', '10105.png', '20175.png', '17790.png', '14883.png', '01490.png', '04759.png', '02090.png', '07438.png', '18095.png', '19821.png', '08903.png', '20252.png', '21939.png', '14936.png', '22617.png', '08801.png', '13355.png', '20423.png', '05077.png', '13419.png', '24856.png', '11892.png', '20061.png', '24580.png', '15697.png', '07096.png', '07098.png', '19622.png', '08155.png', '11604.png', '10749.png', '00141.png', '04693.png', '21050.png', '07442.png', '22706.png', '15273.png', '10352.png', '01125.png', '07279.png', '11363.png', '00366.png', '20604.png', '13335.png', '00226.png', '17352.png', '09851.png', '02817.png', '06049.png', '00199.png', '12803.png', '06678.png', '09478.png', '05799.png', '08360.png', '14062.png', '17878.png', '21477.png', '21452.png', '00908.png', '00277.png', '17333.png', '24117.png', '10261.png', '17960.png', '23428.png', '20692.png', '01915.png', '17773.png', '05436.png', '19102.png', '07482.png', '13219.png', '09963.png', '07153.png', '11359.png', '16837.png', '13129.png', '21582.png', '09218.png', '04669.png', '19366.png', '03953.png', '22217.png', '14608.png', '04744.png', '05086.png', '00752.png', '11635.png', '07910.png', '00929.png', '04353.png', '07189.png', '00089.png', '19524.png', '20432.png', '07448.png', '09314.png', '16683.png', '05594.png', '18077.png', '01784.png', '01267.png', '14491.png', '11813.png', '13758.png', '11045.png', '06675.png', '02480.png', '03664.png', '10654.png', '13248.png', '20711.png', '16764.png', '07493.png', '24306.png', '16034.png', '24029.png', '16210.png', '12071.png', '00936.png', '17946.png', '15825.png', '15693.png', '21476.png', '07535.png', '09969.png', '07235.png', '08287.png', '08076.png', '21947.png', '23484.png', '09552.png', '06500.png', '06708.png', '00262.png', '15409.png', '11268.png', '17683.png', '23701.png', '14623.png', '13839.png', '18192.png', '06892.png', '00845.png', '00612.png', '19981.png', '05678.png', '08463.png', '17168.png', '00361.png', '15333.png', '14890.png', '12321.png', '04027.png', '15592.png', '09508.png', '08723.png', '16841.png', '10575.png', '15798.png', '04458.png', '22922.png', '16335.png', '00492.png', '14420.png', '14444.png', '05322.png', '11544.png', '06462.png', '16948.png', '02142.png', '09733.png', '21616.png', '00241.png', '15181.png', '00682.png', '00543.png', '06219.png', '00902.png', '04825.png', '10459.png', '09118.png', '06478.png', '08095.png', '04088.png', '22114.png', '18390.png', '10975.png', '05231.png', '23116.png', '07832.png', '03635.png', '01941.png', '14065.png', '18832.png', '15398.png', '01867.png', '00528.png', '05749.png', '20376.png', '11867.png', '18841.png', '14943.png', '05616.png', '23965.png', '06034.png', '15385.png', '04492.png', '06463.png', '13871.png', '11748.png', '23977.png', '04380.png', '20199.png', '19259.png', '16528.png', '06700.png', '10109.png', '09896.png', '05884.png', '12917.png', '09081.png', '23194.png', '04516.png', '19293.png', '00179.png', '09351.png', '16903.png', '17307.png', '23614.png', '14185.png', '09378.png', '23375.png', '03645.png', '15799.png', '19291.png', '14774.png', '05051.png', '21786.png', '21507.png', '23863.png', '22737.png', '00460.png', '21317.png', '00507.png', '21192.png', '10658.png', '12271.png', '07942.png', '23432.png', '22008.png', '09167.png', '10849.png', '04039.png', '11034.png', '16876.png', '20489.png', '17742.png', '02685.png', '11130.png', '15706.png', '20352.png', '24585.png', '20188.png', '17463.png', '13526.png', '02557.png', '03544.png', '02342.png', '04399.png', '23970.png', '01261.png', '16097.png', '11250.png', '20808.png', '17032.png', '10453.png', '24078.png', '01535.png', '03509.png', '23406.png', '03846.png', '19587.png', '02461.png', '03797.png', '07776.png', '05174.png', '19030.png', '03356.png', '24778.png', '03359.png', '17353.png', '24003.png', '03904.png', '17293.png', '16618.png', '16333.png', '12646.png', '22995.png', '14759.png', '24058.png', '19554.png', '08070.png', '06150.png', '19057.png', '10114.png', '10806.png', '04847.png', '07148.png', '08634.png', '15547.png', '02094.png', '11379.png', '13019.png', '22891.png', '17091.png', '04282.png', '24467.png', '19905.png', '10400.png', '23435.png', '14668.png', '15023.png', '03348.png', '03119.png', '07965.png', '11573.png', '10508.png', '18520.png', '08100.png', '06453.png', '24313.png', '01768.png', '03903.png', '03691.png', '16056.png', '19775.png', '00152.png', '06936.png', '22960.png', '23263.png', '03661.png', '00827.png', '01485.png', '07246.png', '16769.png', '01023.png', '19230.png', '05289.png', '12619.png', '10348.png', '05379.png', '14539.png', '23535.png', '23239.png', '04856.png', '05451.png', '21839.png', '00364.png', '23826.png', '19682.png', '10321.png', '00646.png', '09932.png', '24603.png', '17757.png', '19541.png', '11270.png', '22733.png', '22933.png', '11876.png', '00290.png', '03479.png', '22273.png', '10544.png', '01517.png', '20170.png', '11102.png', '00398.png', '00251.png', '04387.png', '10351.png', '04192.png', '01731.png', '00926.png', '04161.png', '15858.png', '04738.png', '17518.png', '06910.png', '18108.png', '15964.png', '16224.png', '10323.png', '17313.png', '15683.png', '08897.png', '13430.png', '13726.png', '24750.png', '07667.png', '04904.png', '20450.png', '10528.png', '11500.png', '12772.png', '01245.png', '16684.png', '10823.png', '10086.png', '19797.png', '09577.png', '16390.png', '12883.png', '10191.png', '04082.png', '14301.png', '01560.png', '08086.png', '11899.png', '12821.png', '13347.png', '03464.png', '03386.png', '15787.png', '21348.png', '09889.png', '03486.png', '15518.png', '04672.png', '07001.png', '06897.png', '09230.png', '02043.png', '07314.png', '24756.png', '19717.png', '22712.png', '13015.png', '09594.png', '10638.png', '03211.png', '05265.png', '01333.png', '01288.png', '10433.png', '09102.png', '07693.png', '18404.png', '22760.png', '20714.png', '20116.png', '22182.png', '14563.png', '13770.png', '19510.png', '04564.png', '14830.png', '19337.png', '01901.png', '05130.png', '24223.png', '22431.png', '01258.png', '04695.png', '12499.png', '22595.png', '06277.png', '05527.png', '12498.png', '09397.png', '08911.png', '18564.png', '20774.png', '18376.png', '16852.png', '06844.png', '01370.png', '05635.png', '14938.png', '18248.png', '05684.png', '13986.png', '00282.png', '12703.png', '01735.png', '12653.png', '22748.png', '24315.png', '12310.png', '08183.png', '10325.png', '10599.png', '20193.png', '20283.png', '03772.png', '14320.png', '13372.png', '15807.png', '17102.png', '20816.png', '02797.png', '00256.png', '10074.png', '10605.png', '17351.png', '11804.png', '05172.png', '13553.png', '04100.png', '20806.png', '16590.png', '03087.png', '01034.png', '20971.png', '14742.png', '19344.png', '05514.png', '15288.png', '14487.png', '08043.png', '06352.png', '11920.png', '17060.png', '17241.png', '06544.png', '22254.png', '00260.png', '23668.png', '07759.png', '09453.png', '18947.png', '14954.png', '08498.png', '07644.png', '18985.png', '14810.png', '04427.png', '14730.png', '06509.png', '18826.png', '01754.png', '14127.png', '02743.png', '15533.png', '23642.png', '13147.png', '24385.png', '12162.png', '15521.png', '19273.png', '23501.png', '05298.png', '12156.png', '03975.png', '15227.png', '10777.png', '12391.png', '00916.png', '06051.png', '20231.png', '18245.png', '15488.png', '18592.png', '22492.png', '03957.png', '12633.png', '22647.png', '04050.png', '10773.png', '03489.png', '13031.png', '21138.png', '02750.png', '22463.png', '12167.png', '18411.png', '18476.png', '02474.png', '24121.png', '05732.png', '00671.png', '04789.png', '03236.png', '09320.png', '14331.png', '15349.png', '13733.png', '05281.png', '03140.png', '05106.png', '18679.png', '06464.png', '21668.png', '23010.png', '05025.png', '17336.png', '07921.png', '13066.png', '23897.png', '16706.png', '07245.png', '16493.png', '19462.png', '02160.png', '12958.png', '00503.png', '03450.png', '12095.png', '07209.png', '17300.png', '09460.png', '18004.png', '21373.png', '22040.png', '20443.png', '12428.png', '04344.png', '21286.png', '00130.png', '16043.png', '15183.png', '19438.png', '19946.png', '19530.png', '11331.png', '16755.png', '22247.png', '15007.png', '16395.png', '15489.png', '01706.png', '22282.png', '12324.png', '21419.png', '14248.png', '06308.png', '02679.png', '22210.png', '15711.png', '01409.png', '05004.png', '19869.png', '00276.png', '13417.png', '07857.png', '17969.png', '07539.png', '01345.png', '17019.png', '08557.png', '12183.png', '24845.png', '10589.png', '20693.png', '10027.png', '03810.png', '20841.png', '20203.png', '21113.png', '10796.png', '06663.png', '16914.png', '07703.png', '21717.png', '19788.png', '02502.png', '13021.png', '20468.png', '09477.png', '21751.png', '08946.png', '12808.png', '14919.png', '01727.png', '10336.png', '03817.png', '21873.png', '06125.png', '24276.png', '22854.png', '21351.png', '20932.png', '23451.png', '02119.png', '13275.png', '22323.png', '19777.png', '06139.png', '13200.png', '20370.png', '03576.png', '21397.png', '04249.png', '21572.png', '24749.png', '20568.png', '11262.png', '15648.png', '18459.png', '13862.png', '16131.png', '23583.png', '24706.png', '08340.png', '21328.png', '01634.png', '05092.png', '20073.png', '14435.png', '11907.png', '18909.png', '22059.png', '21999.png', '04318.png', '10075.png', '04284.png', '04973.png', '24332.png', '17226.png', '14520.png', '22142.png', '00479.png', '00833.png', '24732.png', '01648.png', '24592.png', '01818.png', '05897.png', '03414.png', '19993.png', '17169.png', '09870.png', '15153.png', '16724.png', '01021.png', '18276.png', '21965.png', '05219.png', '09144.png', '14798.png', '01201.png', '06194.png', '24026.png', '23987.png', '22775.png', '06867.png', '03239.png', '16849.png', '00167.png', '17696.png', '03430.png', '13906.png', '14333.png', '08172.png', '20414.png', '15717.png', '00379.png', '02624.png', '07652.png', '01104.png', '05716.png', '14199.png', '16999.png', '00609.png', '24133.png', '16569.png', '11077.png', '01150.png', '13256.png', '22185.png', '03138.png', '04283.png', '09581.png', '23825.png', '15044.png', '01951.png', '12118.png', '01821.png', '02814.png', '06557.png', '22257.png', '15144.png', '18039.png', '23339.png', '12462.png', '17062.png', '13087.png', '23327.png', '10701.png', '15306.png', '13139.png', '00777.png', '05801.png', '00941.png', '22664.png', '15261.png', '23923.png', '00096.png', '03907.png', '06265.png', '15403.png', '05571.png', '23200.png', '14388.png', '09608.png', '13820.png', '07783.png', '15428.png', '15031.png', '04844.png', '18368.png', '07087.png', '02240.png', '21341.png', '19617.png', '03641.png', '24934.png', '01677.png', '24817.png', '24635.png', '18394.png', '02788.png', '22669.png', '22583.png', '16862.png', '23560.png', '22600.png', '19631.png', '09394.png', '00683.png', '03219.png', '04327.png', '11803.png', '19214.png', '01928.png', '03579.png', '21088.png', '19838.png', '07133.png', '16872.png', '09556.png', '07420.png', '14910.png', '22356.png', '18749.png', '16947.png', '12409.png', '06172.png', '19628.png', '22398.png', '14973.png', '06090.png', '13473.png', '00308.png', '20265.png', '09679.png', '12226.png', '08359.png', '24649.png', '24330.png', '19742.png', '01090.png', '17138.png', '23584.png', '23791.png', '06803.png', '09878.png', '22575.png', '17660.png', '13106.png', '15771.png', '13681.png', '23719.png', '24404.png', '20496.png', '01776.png', '18060.png', '23442.png', '10961.png', '12550.png', '03088.png', '00806.png', '15292.png', '18314.png', '14738.png', '18141.png', '05813.png', '21986.png', '08499.png', '24739.png', '05472.png', '08318.png', '24822.png', '14524.png', '14985.png', '13966.png', '23573.png', '05041.png', '02082.png', '14549.png', '03448.png', '02818.png', '21362.png', '17448.png', '07272.png', '01787.png', '17009.png', '10053.png', '13709.png', '24850.png', '22920.png', '13055.png', '06842.png', '09910.png', '21979.png', '12932.png', '11596.png', '17194.png', '21247.png', '21387.png', '22129.png', '04128.png', '21037.png', '02307.png', '09064.png', '03900.png', '07526.png', '09722.png', '04557.png', '23448.png', '20672.png', '05347.png', '21292.png', '11368.png', '24628.png', '15896.png', '21886.png', '07687.png', '13067.png', '03625.png', '21927.png', '10084.png', '15528.png', '15365.png', '06089.png', '11300.png', '04712.png', '23333.png', '03513.png', '23420.png', '17076.png', '00947.png', '08636.png', '14159.png', '02609.png', '13547.png', '06553.png', '10159.png', '20854.png', '06014.png', '12137.png', '01334.png', '17125.png', '04067.png', '00820.png', '13313.png', '10782.png', '13062.png', '16774.png', '18793.png', '17349.png', '24546.png', '10745.png', '20833.png', '00281.png', '04517.png', '01665.png', '18064.png', '10657.png', '09305.png', '23934.png', '16262.png', '13590.png', '05296.png', '03800.png', '16800.png', '24708.png', '00548.png', '02945.png', '21117.png', '06283.png', '08571.png', '02560.png', '18432.png', '23752.png', '23628.png', '20795.png', '21199.png', '04321.png', '12997.png', '13500.png', '24292.png', '03390.png', '13747.png', '15478.png', '12898.png', '04359.png', '13435.png', '24483.png', '12959.png', '07937.png', '08529.png', '18421.png', '22307.png', '21355.png', '24565.png', '11603.png', '00685.png', '10179.png', '15089.png', '03805.png', '15876.png', '05717.png', '16847.png', '00994.png', '08862.png', '07202.png', '15682.png', '11421.png', '00350.png', '22700.png', '07041.png', '16515.png', '23244.png', '21599.png', '10097.png', '22656.png', '12933.png', '10564.png', '00666.png', '12173.png', '12144.png', '08819.png', '12521.png', '03629.png', '00248.png', '06108.png', '18787.png', '09667.png', '11943.png', '24174.png', '06135.png', '23745.png', '20240.png', '06248.png', '16574.png', '12624.png', '19003.png', '15071.png', '18213.png', '22372.png', '07216.png', '03568.png', '08460.png', '11351.png', '15760.png', '22581.png', '17401.png', '09710.png', '04131.png', '12169.png', '02861.png', '24632.png', '23376.png', '10924.png', '08937.png', '23817.png', '05062.png', '05177.png', '01530.png', '01130.png', '19564.png', '03771.png', '24765.png', '18698.png', '14779.png', '24867.png', '00342.png', '21926.png', '10845.png', '01324.png', '24700.png', '22788.png', '06929.png', '00449.png', '22491.png', '00340.png', '07142.png', '02637.png', '22589.png', '22881.png', '20341.png', '23181.png', '07182.png', '10356.png', '22758.png', '11843.png', '24863.png', '10287.png', '11436.png', '07251.png', '07959.png', '00702.png', '24355.png', '14268.png', '14528.png', '21149.png', '12142.png', '24399.png', '19258.png', '18492.png', '18518.png', '08110.png', '07992.png', '24379.png', '22404.png', '17587.png', '09487.png', '00092.png', '00951.png', '02261.png', '24912.png', '10926.png', '04753.png', '03995.png', '19992.png', '05311.png', '14577.png', '02061.png', '02015.png', '00069.png', '16386.png', '23889.png', '21659.png', '01679.png', '15699.png', '01454.png', '16422.png', '16883.png', '12365.png', '15976.png', '18292.png', '07427.png', '20984.png', '00963.png', '20495.png', '20440.png', '17361.png', '14674.png', '12723.png', '22015.png', '08848.png', '20804.png', '09402.png', '22719.png', '08078.png', '11669.png', '07771.png', '16130.png', '02793.png', '18541.png', '04655.png', '04781.png', '11426.png', '03784.png', '01967.png', '06714.png', '13485.png', '00472.png', '01730.png', '10646.png', '21372.png', '04886.png', '04204.png', '09186.png', '12184.png', '15739.png', '04251.png', '22916.png', '04911.png', '17224.png', '20281.png', '03747.png', '09009.png', '08974.png', '21201.png', '16524.png', '19251.png', '14281.png', '22808.png', '22191.png', '17843.png', '13132.png', '09888.png', '03407.png', '01359.png', '03157.png', '08206.png', '10454.png', '06768.png', '18088.png', '02472.png', '12764.png', '07520.png', '20392.png', '18387.png', '01305.png', '04439.png', '02254.png', '00938.png', '11636.png', '12056.png', '17149.png', '11793.png', '18014.png', '20923.png', '07909.png', '24655.png', '10533.png', '18893.png', '06399.png', '04382.png', '20005.png', '17604.png', '00196.png', '04668.png', '02846.png', '05073.png', '16573.png', '04998.png', '03403.png', '00439.png', '14327.png', '07073.png', '03168.png', '07437.png', '15381.png', '15242.png', '13975.png', '16968.png', '15206.png', '02370.png', '23151.png', '24077.png', '24143.png', '22899.png', '22680.png', '18580.png', '00589.png', '02243.png', '19686.png', '14982.png', '01710.png', '21672.png', '03909.png', '03176.png', '08219.png', '16009.png', '05600.png', '00125.png', '09291.png', '10673.png', '19299.png', '11579.png', '23262.png', '18862.png', '17376.png', '02754.png', '01463.png', '06711.png', '14395.png', '08111.png', '05439.png', '24457.png', '03505.png', '19805.png', '04240.png', '08613.png', '05639.png', '08725.png', '24015.png', '08228.png', '24767.png', '16307.png', '02189.png', '01186.png', '24028.png', '02025.png', '11214.png', '21495.png', '03000.png', '01777.png', '04059.png', '12581.png', '22430.png', '04112.png', '19139.png', '19360.png', '19731.png', '19649.png', '09646.png', '05142.png', '11667.png', '15443.png', '05676.png', '21882.png', '03054.png', '02705.png', '21967.png', '06558.png', '03706.png', '14021.png', '14720.png', '15283.png', '04213.png', '24929.png', '05853.png', '12607.png', '15537.png', '05713.png', '13465.png', '05764.png', '18819.png', '23545.png', '17370.png', '13799.png', '02401.png', '19842.png', '17898.png', '10118.png', '09209.png', '09250.png', '14660.png', '09871.png', '19928.png', '16984.png', '24616.png', '18022.png', '13854.png', '16585.png', '21290.png', '12230.png', '06871.png', '21731.png', '02869.png', '09986.png', '10077.png', '00724.png', '21530.png', '18562.png', '12088.png', '18854.png', '05171.png', '13047.png', '03591.png', '03814.png', '01181.png', '20447.png', '11697.png', '16779.png', '12967.png', '10122.png', '10818.png', '07928.png', '22229.png', '14535.png', '20621.png', '12874.png', '03632.png', '11126.png', '21712.png', '21313.png', '22451.png', '16902.png', '01142.png', '17217.png', '06367.png', '21732.png', '20955.png', '18065.png', '15098.png', '17452.png', '02767.png', '06539.png', '16424.png', '08391.png', '11190.png', '24100.png', '02809.png', '13623.png', '05411.png', '20476.png', '11170.png', '23969.png', '02963.png', '00610.png', '12398.png', '08882.png', '06699.png', '10985.png', '02594.png', '15814.png', '03596.png', '05693.png', '08294.png', '00006.png', '08966.png', '22982.png', '07364.png', '22927.png', '05325.png', '21677.png', '20356.png', '14215.png', '24168.png', '14116.png', '13040.png', '07668.png', '07508.png', '04542.png', '04879.png', '19522.png', '09430.png', '08549.png', '03241.png', '09203.png', '02139.png', '09809.png', '08622.png', '05166.png', '13848.png', '23186.png', '00797.png', '20434.png', '01462.png', '14889.png', '20633.png', '02908.png', '19839.png', '03228.png', '10909.png', '21933.png', '21053.png', '11617.png', '15419.png', '20793.png', '13350.png', '22638.png', '04159.png', '16976.png', '12006.png', '24928.png', '07028.png', '04786.png', '12682.png', '18309.png', '04762.png', '15531.png', '06279.png', '17646.png', '23623.png', '17414.png', '20596.png', '10481.png', '12274.png', '09654.png', '01198.png', '15143.png', '05138.png', '16302.png', '07408.png', '07890.png', '00631.png', '09618.png', '23143.png', '20561.png', '02644.png', '11160.png', '24415.png', '00569.png', '19232.png', '21940.png', '04315.png', '22621.png', '16973.png', '04243.png', '08268.png', '13633.png', '14499.png', '03773.png', '00315.png', '05259.png', '01081.png', '02360.png', '11247.png', '05156.png', '23877.png', '08887.png', '11389.png', '12070.png', '03169.png', '16183.png', '11398.png', '00323.png', '11316.png', '15483.png', '04602.png', '16802.png', '18830.png', '21771.png', '05000.png', '11288.png', '10611.png', '12794.png', '07285.png', '15401.png', '19531.png', '06036.png', '12132.png', '08880.png', '21831.png', '24348.png', '24838.png', '19222.png', '22450.png', '10152.png', '20338.png', '22662.png', '16565.png', '20666.png', '10685.png', '14580.png', '17732.png', '17440.png', '16128.png', '06299.png', '04331.png', '13591.png', '13236.png', '02255.png', '16482.png', '20746.png', '05397.png', '02020.png', '10651.png', '08959.png', '05490.png', '12395.png', '09917.png', '11204.png', '02329.png', '09308.png', '00381.png', '01072.png', '10038.png', '24659.png', '24588.png', '18093.png', '12714.png', '19431.png', '09816.png', '03146.png', '07917.png', '08010.png', '14849.png', '20051.png', '21848.png', '19475.png', '03829.png', '03312.png', '22153.png', '00633.png', '22977.png', '02520.png', '14818.png', '23326.png', '15865.png', '19578.png', '23622.png', '20146.png', '17912.png', '22533.png', '20136.png', '12296.png', '03115.png', '04125.png', '07524.png', '18401.png', '13985.png', '15113.png', '24164.png', '02935.png', '17942.png', '15260.png', '01210.png', '13700.png', '11925.png', '24897.png', '03736.png', '12443.png', '19620.png', '02891.png', '00572.png', '00063.png', '15598.png', '05955.png', '17746.png', '05232.png', '19927.png', '00674.png', '07849.png', '12126.png', '00595.png', '11770.png', '00831.png', '05911.png', '18172.png', '13901.png', '12722.png', '19822.png', '11590.png', '23958.png', '11909.png', '12776.png', '14297.png', '20331.png', '02677.png', '17588.png', '22842.png', '02840.png', '24031.png', '24773.png', '09866.png', '20658.png', '11315.png', '04352.png', '16887.png', '21105.png', '03933.png', '08575.png', '20605.png', '00132.png', '06037.png', '14556.png', '20049.png', '06523.png', '22936.png', '18817.png', '07483.png', '08566.png', '17301.png', '08170.png', '04675.png', '16751.png', '03294.png', '09480.png', '09861.png', '24356.png', '00349.png', '19191.png', '16860.png', '04737.png', '09907.png', '17830.png', '06346.png', '15328.png', '07706.png', '18600.png', '06077.png', '20110.png', '16941.png', '11817.png', '02384.png', '24609.png', '08545.png', '23976.png', '16737.png', '07602.png', '02658.png', '20533.png', '07291.png', '20204.png', '09001.png', '17172.png', '20551.png', '23874.png', '20360.png', '00486.png', '24302.png', '03139.png', '24175.png', '21358.png', '12021.png', '04661.png', '00506.png', '08225.png', '05199.png', '12341.png', '06608.png', '22537.png', '07241.png', '14704.png', '05685.png', '03561.png', '05212.png', '17058.png', '13262.png', '01158.png', '21205.png', '08608.png', '12891.png', '10279.png', '13496.png', '03560.png', '01703.png', '18471.png', '24526.png', '02348.png', '18494.png', '03891.png', '22836.png', '07949.png', '13745.png', '18364.png', '16153.png', '02392.png', '22573.png', '13499.png', '09447.png', '16052.png', '21619.png', '07611.png', '15900.png', '09204.png', '00863.png', '03592.png', '11354.png', '24614.png', '23215.png', '01858.png', '07718.png', '23577.png', '15779.png', '09160.png', '16170.png', '15120.png', '15654.png', '05934.png', '00445.png', '14942.png', '06011.png', '00770.png', '21589.png', '09526.png', '06777.png', '12503.png', '03195.png', '23744.png', '24127.png', '10418.png', '00694.png', '21815.png', '22561.png', '21600.png', '24027.png', '24960.png', '06497.png', '05081.png', '17047.png', '07120.png', '09170.png', '01888.png', '19422.png', '16992.png', '08664.png', '15373.png', '01740.png', '08531.png', '01670.png', '07504.png', '01270.png', '15251.png', '11458.png', '13703.png', '01030.png', '12489.png', '16353.png', '08044.png', '22062.png', '14895.png', '05016.png', '23840.png', '24920.png', '07972.png', '18952.png', '18220.png', '10091.png', '11999.png', '04316.png', '07904.png', '01921.png', '16305.png', '02371.png', '06821.png', '03511.png', '21761.png', '24913.png', '06950.png', '07310.png', '13634.png', '14106.png', '18026.png', '22791.png', '00140.png', '20690.png', '03972.png', '23196.png', '01502.png', '08446.png', '09133.png', '22671.png', '16338.png', '02890.png', '15316.png', '10530.png', '03890.png', '19555.png', '22631.png', '20491.png', '13949.png', '01050.png', '02408.png', '06735.png', '16344.png', '10253.png', '17686.png', '17975.png', '06001.png', '12356.png', '04796.png', '07260.png', '23846.png', '15156.png', '03060.png', '09051.png', '02885.png', '07184.png', '20320.png', '19511.png', '22014.png', '16070.png', '22918.png', '06262.png', '07357.png', '06236.png', '09264.png', '00574.png', '24744.png', '03280.png', '21148.png', '12152.png', '04722.png', '04094.png', '11946.png', '02859.png', '14377.png', '11469.png', '18803.png', '24335.png', '22304.png', '17778.png', '09957.png', '06247.png', '23220.png', '03079.png', '00991.png', '24931.png', '08813.png', '07292.png', '22496.png', '21163.png', '23291.png', '14229.png', '13761.png', '23899.png', '17980.png', '09114.png', '21161.png', '17549.png', '17749.png', '16164.png', '10524.png', '03097.png', '17846.png', '10565.png', '07208.png', '13472.png', '18521.png', '14828.png', '07906.png', '11438.png', '00566.png', '00176.png', '15205.png', '07352.png', '05652.png', '14142.png', '11085.png', '16720.png', '15294.png', '05474.png', '09682.png', '02224.png', '17578.png', '12351.png', '18605.png', '13268.png', '17075.png', '11585.png', '14607.png', '17858.png', '19441.png', '16061.png', '09664.png', '03811.png', '20359.png', '20624.png', '00046.png', '19211.png', '00696.png', '23590.png', '14005.png', '11574.png', '11055.png', '08967.png', '11541.png', '23088.png', '07558.png', '17884.png', '04763.png', '04228.png', '19331.png', '00975.png', '13315.png', '23476.png', '22619.png', '05075.png', '03482.png', '05415.png', '23672.png', '02122.png', '00405.png', '18561.png', '00943.png', '22478.png', '03422.png', '18139.png', '01410.png', '14452.png', '23429.png', '06824.png', '02432.png', '13000.png', '22053.png', '12706.png', '12701.png', '21827.png', '06802.png', '12468.png', '23334.png', '03245.png', '16409.png', '09914.png', '20098.png', '18965.png', '24189.png', '03567.png', '19596.png', '12756.png', '03061.png', '03096.png', '13918.png', '24697.png', '08090.png', '07582.png', '16832.png', '03521.png', '11622.png', '23996.png', '03659.png', '01752.png', '20101.png', '19446.png', '05143.png', '17443.png', '17823.png', '21687.png', '01357.png', '20924.png', '13686.png', '14461.png', '17214.png', '22630.png', '05510.png', '01337.png', '08867.png', '06885.png', '06476.png', '02899.png', '24501.png', '04640.png', '06506.png', '15866.png', '06820.png', '06665.png', '10678.png', '16675.png', '20744.png', '01842.png', '20675.png', '07836.png', '08522.png', '02405.png', '15651.png', '10537.png', '24155.png', '21938.png', '24768.png', '20754.png', '19397.png', '01313.png', '20822.png', '05603.png', '19895.png', '07309.png', '10488.png', '06264.png', '03478.png', '11983.png', '16744.png', '23747.png', '17582.png', '04051.png', '18447.png', '11852.png', '14912.png', '06585.png', '00638.png', '00186.png', '14122.png', '22710.png', '09443.png', '15191.png', '10719.png', '21743.png', '14995.png', '15937.png', '15974.png', '23197.png', '15195.png', '09595.png', '17883.png', '18021.png', '05116.png', '21661.png', '19420.png', '16380.png', '00114.png', '14187.png', '02736.png', '20831.png', '06029.png', '09152.png', '17617.png', '24879.png', '08009.png', '04021.png', '07856.png', '22432.png', '20205.png', '07497.png', '15642.png', '10632.png', '08924.png', '14035.png', '11789.png', '15669.png', '09006.png', '13680.png', '23241.png', '02505.png', '13394.png', '05476.png', '12077.png', '04374.png', '15201.png', '05519.png', '04690.png', '09848.png', '19351.png', '23675.png', '12977.png', '17444.png', '02583.png', '15677.png', '11192.png', '19880.png', '17772.png', '06646.png', '13980.png', '16580.png', '21091.png', '23984.png', '08553.png', '01557.png', '11032.png', '08753.png', '02456.png', '20085.png', '03677.png', '01970.png', '20582.png', '23827.png', '21156.png', '18640.png', '03286.png', '04694.png', '16985.png', '24336.png', '03633.png', '23621.png', '13301.png', '03950.png', '06649.png', '17466.png', '15951.png', '20558.png', '00123.png', '22525.png', '01936.png', '09940.png', '00207.png', '02246.png', '06297.png', '23370.png', '00842.png', '03855.png', '22473.png', '24781.png', '12735.png', '09139.png', '09950.png', '02779.png', '06783.png', '19553.png', '04338.png', '24956.png', '02345.png', '21375.png', '17148.png', '02923.png', '10273.png', '22861.png', '17178.png', '19693.png', '22488.png', '03847.png', '13479.png', '01349.png', '05017.png', '24420.png', '21930.png', '16697.png', '00163.png', '01861.png', '13917.png', '08906.png', '06032.png', '06185.png', '12536.png', '17484.png', '15811.png', '10645.png', '22542.png', '15442.png', '05674.png', '07368.png', '07889.png', '20866.png', '13569.png', '08986.png', '15890.png', '18104.png', '10144.png', '01556.png', '01495.png', '24172.png', '00376.png', '05651.png', '11028.png', '08473.png', '18828.png', '22738.png', '20916.png', '24495.png', '10973.png', '07528.png', '01590.png', '08313.png', '06257.png', '05357.png', '06583.png', '10297.png', '03043.png', '03929.png', '15027.png', '06595.png', '12667.png', '04620.png', '19032.png', '05773.png', '02454.png', '04750.png', '22796.png', '19845.png', '12914.png', '05082.png', '00213.png', '01610.png', '22925.png', '12029.png', '05555.png', '19671.png', '22586.png', '24729.png', '04455.png', '08424.png', '17706.png', '03231.png', '05735.png', '05818.png', '01678.png', '15686.png', '07806.png', '08501.png', '14422.png', '12518.png', '02826.png', '05387.png', '14437.png', '12636.png', '09121.png', '17872.png', '17820.png', '07768.png', '18308.png', '01508.png', '10429.png', '11405.png', '23249.png', '15612.png', '02200.png', '02144.png', '01299.png', '15123.png', '13718.png', '04457.png', '01255.png', '10154.png', '21064.png', '09196.png', '17277.png', '13316.png', '02881.png', '12643.png', '04945.png', '01405.png', '22722.png', '00882.png', '16666.png', '02642.png', '07754.png', '16485.png', '08767.png', '08562.png', '14939.png', '06437.png', '19957.png', '13782.png', '11413.png', '02486.png', '20716.png', '21326.png', '20267.png', '02463.png', '04160.png', '17707.png', '14086.png', '14712.png', '01588.png', '04587.png', '18945.png', '03913.png', '21171.png', '24802.png', '13396.png', '02093.png', '20954.png', '03681.png', '11267.png', '16388.png', '21298.png', '03112.png', '19190.png', '09658.png', '17192.png', '13312.png', '24853.png', '11083.png', '19266.png', '06730.png', '09084.png', '08918.png', '17863.png', '03783.png', '10407.png', '12181.png', '04740.png', '18663.png', '12987.png', '18624.png', '07125.png', '16273.png', '02748.png', '23383.png', '18899.png', '00807.png', '24530.png', '15168.png', '20896.png', '02302.png', '22122.png', '01533.png', '09420.png', '10326.png', '13242.png', '22961.png', '18047.png', '11243.png', '21354.png', '02427.png', '17900.png', '12592.png', '20094.png', '03440.png', '03094.png', '16639.png', '02397.png', '15269.png', '14555.png', '15441.png', '07063.png', '12992.png', '11777.png', '09256.png', '10902.png', '23586.png', '06994.png', '08777.png', '05200.png', '21307.png', '04044.png', '11749.png', '14203.png', '00030.png', '14001.png', '07518.png', '14817.png', '24103.png', '09089.png', '13328.png', '15908.png', '23574.png', '14915.png', '12886.png', '17180.png', '11162.png', '06684.png', '02916.png', '04718.png', '11151.png', '06840.png', '17629.png', '19433.png', '19975.png', '14707.png', '11882.png', '18469.png', '13290.png', '13414.png', '01959.png', '14034.png', '21115.png', '05315.png', '10145.png', '09151.png', '00491.png', '23948.png', '16808.png', '12586.png', '14179.png', '11647.png', '06487.png', '02316.png', '04730.png', '15406.png', '21182.png', '08539.png', '21897.png', '16609.png', '08402.png', '05210.png', '21526.png', '20238.png', '13560.png', '20545.png', '06813.png', '22487.png', '02417.png', '01358.png', '07262.png', '06048.png', '09553.png', '04972.png', '12815.png', '23756.png', '18809.png', '07738.png', '14797.png', '01317.png', '17961.png', '20967.png', '22633.png', '18018.png', '13134.png', '19107.png', '17103.png', '09225.png', '08156.png', '17521.png', '08883.png', '04645.png', '11688.png', '08335.png', '13214.png', '00709.png', '22839.png', '04422.png', '00090.png', '00504.png', '02599.png', '18807.png', '01996.png', '03363.png', '21780.png', '18448.png', '24917.png', '07697.png', '19800.png', '08888.png', '11093.png', '20181.png', '19326.png', '15200.png', '09368.png', '16445.png', '02386.png', '04846.png', '05826.png', '08436.png', '11931.png', '18059.png', '00825.png', '03341.png', '00824.png', '19238.png', '02321.png', '17036.png', '13477.png', '22325.png', '20016.png', '06442.png', '22704.png', '11625.png', '02973.png', '21699.png', '02138.png', '07071.png', '03293.png', '23830.png', '19339.png', '06814.png', '10505.png', '14944.png', '05807.png', '17911.png', '02636.png', '00570.png', '07134.png', '18783.png', '00391.png', '19375.png', '15972.png', '01321.png', '21458.png', '11913.png', '23912.png', '12075.png', '05726.png', '19151.png', '04023.png', '01406.png', '12372.png', '08274.png', '21915.png', '05317.png', '16093.png', '23269.png', '00303.png', '19026.png', '15678.png', '01769.png', '22297.png', '07168.png', '14184.png', '05637.png', '04391.png', '22489.png', '06965.png', '03778.png', '05841.png', '18623.png', '19076.png', '19135.png', '04927.png', '15880.png', '17316.png', '06918.png', '21598.png', '21775.png', '02638.png', '14276.png', '21025.png', '06080.png', '10941.png', '19053.png', '17805.png', '02713.png', '22301.png', '06093.png', '23310.png', '19921.png', '20388.png', '16000.png', '17890.png', '18127.png', '05170.png', '17774.png', '05565.png', '03994.png', '16158.png', '21601.png', '03910.png', '20953.png', '10875.png', '05345.png', '13568.png', '03643.png', '24449.png', '05479.png', '17276.png', '08152.png', '12081.png', '12287.png', '03604.png', '13059.png', '17426.png', '19312.png', '12771.png', '21133.png', '17396.png', '10797.png', '23311.png', '22707.png', '18617.png', '15408.png', '15075.png', '03858.png', '12270.png', '24272.png', '14860.png', '13615.png', '18445.png', '24384.png', '22530.png', '11246.png', '15286.png', '23364.png', '00149.png', '04191.png', '11910.png', '00603.png', '16364.png', '14663.png', '14460.png', '17133.png', '01360.png', '07495.png', '22480.png', '06501.png', '22482.png', '01411.png', '01646.png', '23783.png', '07237.png', '15347.png', '19271.png', '04024.png', '02465.png', '14532.png', '19489.png', '01474.png', '07752.png', '10742.png', '17526.png', '03215.png', '09522.png', '10121.png', '22856.png', '03411.png', '01016.png', '21461.png', '22979.png', '13514.png', '09388.png', '13638.png', '19062.png', '01877.png', '04505.png', '08301.png', '15955.png', '09541.png', '11016.png', '03538.png', '18536.png', '11159.png', '10919.png', '02225.png', '04697.png', '17244.png', '09205.png', '16053.png', '08696.png', '13662.png', '01800.png', '20521.png', '17249.png', '06017.png', '21767.png', '08072.png', '07880.png', '24359.png', '22295.png', '21552.png', '03106.png', '09069.png', '21255.png', '21909.png', '14295.png', '09410.png', '07948.png', '13930.png', '07358.png', '24373.png', '07819.png', '23856.png', '03263.png', '11665.png', '00946.png', '05348.png', '14263.png', '07915.png', '03400.png', '24574.png', '23300.png', '07797.png', '19740.png', '20036.png', '21359.png', '12407.png', '21678.png', '24862.png', '05475.png', '09808.png', '15494.png', '04820.png', '05656.png', '16935.png', '22931.png', '14694.png', '20454.png', '24150.png', '14085.png', '24806.png', '19943.png', '18710.png', '15650.png', '22168.png', '14881.png', '12807.png', '06013.png', '18674.png', '19083.png', '19383.png', '07534.png', '24034.png', '21637.png', '04862.png', '15291.png', '05959.png', '24829.png', '19867.png', '15579.png', '24629.png', '21279.png', '14726.png', '08717.png', '23170.png', '11747.png', '06788.png', '19730.png', '05310.png', '24625.png', '24578.png', '11113.png', '15758.png', '13382.png', '07985.png', '00791.png', '05589.png', '04968.png', '15534.png', '21742.png', '24718.png', '22434.png', '00830.png', '07846.png', '03725.png', '00861.png', '21172.png', '09783.png', '17862.png', '20131.png', '18753.png', '10896.png', '11033.png', '02374.png', '00261.png', '10136.png', '20847.png', '08958.png', '13291.png', '01449.png', '22694.png', '20645.png', '14084.png', '14879.png', '19792.png', '04335.png', '05211.png', '17917.png', '06224.png', '24408.png', '24346.png', '14541.png', '06668.png', '06838.png', '01277.png', '13668.png', '05805.png', '20878.png', '02541.png', '03687.png', '02712.png', '20405.png', '09073.png', '18120.png', '12378.png', '13605.png', '03167.png', '20721.png', '14751.png', '23299.png', '10283.png', '17800.png', '08061.png', '18480.png', '02230.png', '22532.png', '17528.png', '10629.png', '19253.png', '13821.png', '17841.png', '18019.png', '17577.png', '05065.png', '15340.png', '09323.png', '04724.png', '14741.png', '16438.png', '19890.png', '05707.png', '04096.png', '04055.png', '19685.png', '03867.png', '24266.png', '14432.png', '24643.png', '13229.png', '18603.png', '13217.png', '02665.png', '16689.png', '00432.png', '02069.png', '14566.png', '02834.png', '14763.png', '22027.png', '02964.png', '19514.png', '22437.png', '12991.png', '12053.png', '15131.png', '05721.png', '00541.png', '06732.png', '12926.png', '22629.png', '14052.png', '23520.png', '08688.png', '24001.png', '03132.png', '19169.png', '23394.png', '01472.png', '24557.png', '22126.png', '04144.png', '09371.png', '06302.png', '19359.png', '00795.png', '05306.png', '15072.png', '15073.png', '10860.png', '16804.png', '05922.png', '06040.png', '02045.png', '19325.png', '19864.png', '24836.png', '22867.png', '09582.png', '07624.png', '05808.png', '18883.png', '08482.png', '04358.png', '02334.png', '14605.png', '05592.png', '05647.png', '21365.png', '09764.png', '12438.png', '07425.png', '08930.png', '21800.png', '07996.png', '13478.png', '14186.png', '20102.png', '15137.png', '23853.png', '00210.png', '02616.png', '20227.png', '22627.png', '20790.png', '08123.png', '08393.png', '04533.png', '04803.png', '09135.png', '11062.png', '08135.png', '05250.png', '11207.png', '14077.png', '06458.png', '12392.png', '06890.png', '09964.png', '19673.png', '04888.png', '21959.png', '20243.png', '02663.png', '20534.png', '15657.png', '08704.png', '13267.png', '09634.png', '10318.png', '12170.png', '20302.png', '02038.png', '22935.png', '19492.png', '13896.png', '00268.png', '17780.png', '22072.png', '11026.png', '03446.png', '02977.png', '01567.png', '05255.png', '10361.png', '08901.png', '20914.png', '16727.png', '11406.png', '11862.png', '23360.png', '09154.png', '15676.png', '01647.png', '24423.png', '09361.png', '12612.png', '21698.png', '22865.png', '15186.png', '08055.png', '10123.png', '14578.png', '12781.png', '07792.png', '19840.png', '05015.png', '10570.png', '05445.png', '00475.png', '11519.png', '11138.png', '18355.png', '12273.png', '13221.png', '11540.png', '16912.png', '15433.png', '07039.png', '10256.png', '16138.png', '08176.png', '07083.png', '20928.png', '21937.png', '24820.png', '08037.png', '12564.png', '00912.png', '04086.png', '20981.png', '15218.png', '16851.png', '01895.png', '19523.png', '06359.png', '17944.png', '03754.png', '15309.png', '18869.png', '02311.png', '21414.png', '17665.png', '23472.png', '19868.png', '13853.png', '05352.png', '09137.png', '09843.png', '02400.png', '17612.png', '04863.png', '05642.png', '01397.png', '15987.png', '23713.png', '17338.png', '21735.png', '05187.png', '11001.png', '06661.png', '24309.png', '21571.png', '16177.png', '12748.png', '14110.png', '00358.png', '20492.png', '16818.png', '08366.png', '23101.png', '14854.png', '17266.png', '22577.png', '02958.png', '15093.png', '07607.png', '14740.png', '00080.png', '03285.png', '08879.png', '01833.png', '19837.png', '20126.png', '02475.png', '19769.png', '00699.png', '20510.png', '20708.png', '04959.png', '00580.png', '22607.png', '11295.png', '13330.png', '19152.png', '21143.png', '07424.png', '06278.png', '19267.png', '24842.png', '23421.png', '14684.png', '14815.png', '16669.png', '01467.png', '24032.png', '03822.png', '13298.png', '06607.png', '18428.png', '09521.png', '06766.png', '10908.png', '15368.png', '21024.png', '11320.png', '02369.png', '22941.png', '13622.png', '06568.png', '09734.png', '14196.png', '01352.png', '18232.png', '08355.png', '11944.png', '13994.png', '13302.png', '08140.png', '17357.png', '02146.png', '14030.png', '07188.png', '04550.png', '17958.png', '03257.png', '14559.png', '03117.png', '13071.png', '04019.png', '05204.png', '10049.png', '21220.png', '23746.png', '19031.png', '00517.png', '20859.png', '13522.png', '11981.png', '11430.png', '11384.png', '08314.png', '24304.png', '10272.png', '14796.png', '13505.png', '18360.png', '07851.png', '04185.png', '07203.png', '22756.png', '01954.png', '00025.png', '01636.png', '02309.png', '08073.png', '15042.png', '09662.png', '18905.png', '09125.png', '16343.png', '10240.png', '19485.png', '23294.png', '06979.png', '05914.png', '07480.png', '14689.png', '11087.png', '21219.png', '13348.png', '15775.png', '07135.png', '23847.png', '13897.png', '19713.png', '05756.png', '03796.png', '12326.png', '08982.png', '04409.png', '22131.png', '11783.png', '15790.png', '06424.png', '07590.png', '09054.png', '08864.png', '08157.png', '03719.png', '22119.png', '22917.png', '19343.png', '10357.png', '19093.png', '14645.png', '05664.png', '17876.png', '11306.png', '13238.png', '06398.png', '08348.png', '15297.png', '00135.png', '22207.png', '07093.png', '21643.png', '15048.png', '13849.png', '22635.png', '02251.png', '06189.png', '23805.png', '20190.png', '22230.png', '21038.png', '20706.png', '09583.png', '17767.png', '19069.png', '04195.png', '12490.png', '03171.png', '13480.png', '23639.png', '12538.png', '00577.png', '13968.png', '07007.png', '08858.png', '13265.png', '01609.png', '23901.png', '14315.png', '17294.png', '13494.png', '05714.png', '09989.png', '08011.png', '13742.png', '08730.png', '08376.png', '20710.png', '24674.png', '11955.png', '10470.png', '03045.png', '17754.png', '19008.png', '24253.png', '15704.png', '03977.png', '16083.png', '11601.png', '19636.png', '19551.png', '15950.png', '00427.png', '19968.png', '20783.png', '04659.png', '02954.png', '05484.png', '02036.png', '01913.png', '06704.png', '22956.png', '11615.png', '24279.png', '16203.png', '06191.png', '02504.png', '07169.png', '15019.png', '09251.png', '24126.png', '10120.png', '14096.png', '02024.png', '19793.png', '04014.png', '05083.png', '23532.png', '04984.png', '00193.png', '23992.png', '00933.png', '14739.png', '17223.png', '00101.png', '16469.png', '21178.png', '04853.png', '04449.png', '15527.png', '01525.png', '18304.png', '09345.png', '15343.png', '05499.png', '20694.png', '03224.png', '14805.png', '03296.png', '17256.png', '10765.png', '03717.png', '01796.png', '18700.png', '01296.png', '05180.png', '16844.png', '05382.png', '11648.png', '24131.png', '23919.png', '08275.png', '12367.png', '02510.png', '20700.png', '14972.png', '21578.png', '07221.png', '23275.png', '24693.png', '05444.png', '18619.png', '03703.png', '17809.png', '10785.png', '07472.png', '03355.png', '16665.png', '22395.png', '24250.png', '03716.png', '16054.png', '20349.png', '10500.png', '09276.png', '01932.png', '07990.png', '12404.png', '24733.png', '22066.png', '03792.png', '24759.png', '09172.png', '12940.png', '06987.png', '20328.png', '22026.png', '24159.png', '18939.png', '20687.png', '03027.png', '06056.png', '08398.png', '22945.png', '01362.png', '08600.png', '16375.png', '21714.png', '16215.png', '03091.png', '06895.png', '00809.png', '14068.png', '12789.png', '03011.png', '10619.png', '16253.png', '05207.png', '15464.png', '08413.png', '08425.png', '13843.png', '03718.png', '07981.png', '14446.png', '04589.png', '22495.png', '23956.png', '17104.png', '19022.png', '06028.png', '13641.png', '03173.png', '02423.png', '06876.png', '02056.png', '17418.png', '15158.png', '08027.png', '22637.png', '17836.png', '14371.png', '19544.png', '09982.png', '10041.png', '23052.png', '13476.png', '07671.png', '15222.png', '21785.png', '13962.png', '09578.png', '22544.png', '24515.png', '06510.png', '03047.png', '22937.png', '12002.png', '20309.png', '23272.png', '16264.png', '00195.png', '06652.png', '02033.png', '09103.png', '13185.png', '10866.png', '17517.png', '10791.png', '23991.png', '16497.png', '16048.png', '06474.png', '10100.png', '00917.png', '12436.png', '24251.png', '00713.png', '14876.png', '18998.png', '17891.png', '03593.png', '14838.png', '10595.png', '03801.png', '01578.png', '23362.png', '19726.png', '12894.png', '09638.png', '02578.png', '23779.png', '00971.png', '03298.png', '20367.png', '11146.png', '18050.png', '16648.png', '16498.png', '14454.png', '01644.png', '18739.png', '19011.png', '18299.png', '20649.png', '17551.png', '20056.png', '24137.png', '20983.png', '05861.png', '01713.png', '18995.png', '13791.png', '03899.png', '08023.png', '17733.png', '08280.png', '03453.png', '24594.png', '16283.png', '11371.png', '05929.png', '15593.png', '00174.png', '24914.png', '02775.png', '14850.png', '06660.png', '08145.png', '15118.png', '07399.png', '05230.png', '10972.png', '01712.png', '17630.png', '23800.png', '04093.png', '05909.png', '19979.png', '24439.png', '01960.png', '19245.png', '16337.png', '19590.png', '13925.png', '23306.png', '18273.png', '07902.png', '02450.png', '06825.png', '05618.png', '02398.png', '02074.png', '24782.png', '12641.png', '01065.png', '21590.png', '14211.png', '11066.png', '23198.png', '05339.png', '11857.png', '04032.png', '18046.png', '07778.png', '02453.png', '22479.png', '12063.png', '01455.png', '16316.png', '23287.png', '13232.png', '09451.png', '21968.png', '23304.png', '18136.png', '17988.png', '17935.png', '13333.png', '11260.png', '18149.png', '11298.png', '11630.png', '16231.png', '16640.png', '19765.png', '00076.png', '22296.png', '07040.png', '00495.png', '03883.png', '05605.png', '14924.png', '02373.png', '21040.png', '20319.png', '12844.png', '14709.png', '21274.png', '03100.png', '11689.png', '15338.png', '14013.png', '17997.png', '08002.png', '24067.png', '23696.png', '05085.png', '06992.png', '10978.png', '13527.png', '07423.png', '07818.png', '23190.png', '05542.png', '22048.png', '17216.png', '14029.png', '12833.png', '02100.png', '17221.png', '19405.png', '17826.png', '10216.png', '01885.png', '18931.png', '22811.png', '16412.png', '15378.png', '13010.png', '12318.png', '19836.png', '06254.png', '04706.png', '14385.png', '06175.png', '23552.png', '15740.png', '09104.png', '24254.png', '08336.png', '21061.png', '00781.png', '21556.png', '17347.png', '04756.png', '04063.png', '07603.png', '10477.png', '06845.png', '07701.png', '18634.png', '21801.png', '20345.png', '04355.png', '07594.png', '15268.png', '07476.png', '22255.png', '07454.png', '19782.png', '01558.png', '17752.png', '14812.png', '02675.png', '13759.png', '24509.png', '19885.png', '11251.png', '20173.png', '01398.png', '14946.png', '05593.png', '01838.png', '19118.png', '15199.png', '23325.png', '19591.png', '00098.png', '08726.png', '18240.png', '19906.png', '15556.png', '24582.png', '02216.png', '23926.png', '00859.png', '15713.png', '05738.png', '10382.png', '23938.png', '11432.png', '11231.png', '08220.png', '11652.png', '22955.png', '24907.png', '24738.png', '19518.png', '06019.png', '10012.png', '18786.png', '00279.png', '02055.png', '22727.png', '16361.png', '13042.png', '21339.png', '16546.png', '00598.png', '23020.png', '09496.png', '07584.png', '02983.png', '01995.png', '17595.png', '09856.png', '16892.png', '06590.png', '19500.png', '14376.png', '11242.png', '01956.png', '03028.png', '02273.png', '20154.png', '21763.png', '02086.png', '20875.png', '17787.png', '22711.png', '15476.png', '18789.png', '01909.png', '10911.png', '16970.png', '15104.png', '15135.png', '10583.png', '17303.png', '01575.png', '16606.png', '07809.png', '05175.png', '12187.png', '24962.png', '23248.png', '01020.png', '17924.png', '07331.png', '00128.png', '05392.png', '16557.png', '03357.png', '00438.png', '06392.png', '18144.png', '12566.png', '00868.png', '04203.png', '11483.png', '15081.png', '01017.png', '22477.png', '22350.png', '19148.png', '19759.png', '22801.png', '13924.png', '08423.png', '21577.png', '17429.png', '00524.png', '02601.png', '21628.png', '14238.png', '18618.png', '24510.png', '21078.png', '24935.png', '05687.png', '05993.png', '09860.png', '16628.png', '18501.png', '22829.png', '01268.png', '04823.png', '14624.png', '00151.png', '22462.png', '18593.png', '08747.png', '09486.png', '00204.png', '13816.png', '15204.png', '18764.png', '04980.png', '01999.png', '10984.png', '06969.png', '18932.png', '18358.png', '13763.png', '22750.png', '17326.png', '07788.png', '13670.png', '13052.png', '02482.png', '03191.png', '19778.png', '16347.png', '11443.png', '21066.png', '11896.png', '21273.png', '21795.png', '03420.png', '09972.png', '22501.png', '14399.png', '24394.png', '20743.png', '10865.png', '16071.png', '13574.png', '12109.png', '04784.png', '19111.png', '13587.png', '03344.png', '11099.png', '17142.png', '04611.png', '18264.png', '00921.png', '07406.png', '03200.png', '21538.png', '17793.png', '04918.png', '06144.png', '22031.png', '23279.png', '09880.png', '08019.png', '03361.png', '10076.png', '06419.png', '12331.png', '18160.png', '03502.png', '12016.png', '10398.png', '10748.png', '01376.png', '19915.png', '01199.png', '00779.png', '07762.png', '18856.png', '03663.png', '17115.png', '12893.png', '07553.png', '02490.png', '19613.png', '05249.png', '24644.png', '15688.png', '03992.png', '21799.png', '06190.png', '04104.png', '19384.png', '20530.png', '02539.png', '08310.png', '19630.png', '04899.png', '10653.png', '00497.png', '22331.png', '17747.png', '18261.png', '22751.png', '14615.png', '23748.png', '00382.png', '12107.png', '04872.png', '17335.png', '11491.png', '17691.png', '17677.png', '07027.png', '22523.png', '18251.png', '01552.png', '06295.png', '13798.png', '09499.png', '16693.png', '24595.png', '23093.png', '22373.png', '07758.png', '08556.png', '05425.png', '08489.png', '15930.png', '19861.png', '08963.png', '01947.png', '21832.png', '03971.png', '10996.png', '03154.png', '10434.png', '06151.png', '07194.png', '10458.png', '15379.png', '13362.png', '19255.png', '23941.png', '17442.png', '16299.png', '17255.png', '12482.png', '21632.png', '15553.png', '13841.png', '21560.png', '23933.png', '09414.png', '24576.png', '07655.png', '03374.png', '06004.png', '10169.png', '09996.png', '08515.png', '05834.png', '08781.png', '05623.png', '19487.png', '09312.png', '21963.png', '12627.png', '07145.png', '12727.png', '13813.png', '05941.png', '10622.png', '07228.png', '07468.png', '01738.png', '12155.png', '24740.png', '00177.png', '00023.png', '16281.png', '03274.png', '08400.png', '05792.png', '05759.png', '05660.png', '12463.png', '24220.png', '24774.png', '02883.png', '19042.png', '22324.png', '00397.png', '18330.png', '15687.png', '20419.png', '10230.png', '18644.png', '12862.png', '23836.png', '16332.png', '13011.png', '16772.png', '11397.png', '13437.png', '08260.png', '02985.png', '19607.png', '23250.png', '08500.png', '18409.png', '07764.png', '18590.png', '06347.png', '12264.png', '00925.png', '10700.png', '12677.png', '21869.png', '14931.png', '14874.png', '16416.png', '21539.png', '18262.png', '08828.png', '12621.png', '18189.png', '10752.png', '16348.png', '09306.png', '16290.png', '21194.png', '20776.png', '13740.png', '12402.png', '13827.png', '03675.png', '08270.png', '07657.png', '01005.png', '09448.png', '07308.png', '02493.png', '11225.png', '06327.png', '22735.png', '19365.png', '23388.png', '24139.png', '22877.png', '12634.png', '11751.png', '07114.png', '22115.png', '13876.png', '03492.png', '13423.png', '09041.png', '09622.png', '17796.png', '03472.png', '09290.png', '23357.png', '23078.png', '14975.png', '14955.png', '11767.png', '14997.png', '03017.png', '09431.png', '06413.png', '01663.png', '17682.png', '20274.png', '14040.png', '17456.png', '04123.png', '14950.png', '16533.png', '09219.png', '11582.png', '18137.png', '19363.png', '02572.png', '23321.png', '21494.png', '17286.png', '07571.png', '07441.png', '07158.png', '23862.png', '16777.png', '13444.png', '02234.png', '02907.png', '07580.png', '15216.png', '20261.png', '20220.png', '15207.png', '16055.png', '18581.png', '17342.png', '05251.png', '13387.png', '15185.png', '12227.png', '03201.png', '21860.png', '10379.png', '05580.png', '22061.png', '08764.png', '00103.png', '14731.png', '20430.png', '18121.png', '02895.png', '12717.png', '05084.png', '02204.png', '03854.png', '14776.png', '22548.png', '23399.png', '22116.png', '18293.png', '01157.png', '22684.png', '19702.png', '03005.png', '21076.png', '05638.png', '20313.png', '10170.png', '18010.png', '05447.png', '11926.png', '11387.png', '19794.png', '01847.png', '00110.png', '16981.png', '03830.png', '23192.png', '05122.png', '24549.png', '24240.png', '09610.png', '17089.png', '19538.png', '08058.png', '16518.png', '21955.png', '09948.png', '00613.png', '10066.png', '00502.png', '20767.png', '15603.png', '07238.png', '14533.png', '13163.png', '12760.png', '22971.png', '04810.png', '20559.png', '22528.png', '08774.png', '23699.png', '17469.png', '12390.png', '16387.png', '20211.png', '20740.png', '14169.png', '13370.png', '03435.png', '23816.png', '02635.png', '23851.png', '22227.png', '03962.png', '00742.png', '12787.png', '19145.png', '14609.png', '22212.png', '18876.png', '09070.png', '14335.png', '12700.png', '00755.png', '13475.png', '19092.png', '16500.png', '01366.png', '22849.png', '18635.png', '04583.png', '12092.png', '08652.png', '19378.png', '22874.png', '13725.png', '10666.png', '12424.png', '23687.png', '21262.png', '01153.png', '08437.png', '10203.png', '06101.png', '22606.png', '06680.png', '22744.png', '02288.png', '05547.png', '02918.png', '20048.png', '03815.png', '05563.png', '10327.png', '10914.png', '19971.png', '06676.png', '00599.png', '22178.png', '05246.png', '04181.png', '11011.png', '14749.png', '23259.png', '14198.png', '14767.png', '09543.png', '12474.png', '10971.png', '17569.png', '02013.png', '11507.png', '04642.png', '24232.png', '05882.png', '22433.png', '17688.png', '18959.png', '09707.png', '14453.png', '13903.png', '09520.png', '23855.png', '13630.png', '05272.png', '13401.png', '24259.png', '04983.png', '23609.png', '01650.png', '02266.png', '15519.png', '24190.png', '09012.png', '15914.png', '18831.png', '04129.png', '12059.png', '23405.png', '14182.png', '11271.png', '19196.png', '12567.png', '18882.png', '18843.png', '20937.png', '16782.png', '06520.png', '11730.png', '13063.png', '22349.png', '10007.png', '02753.png', '16033.png', '21013.png', '18874.png', '16745.png', '20391.png', '21563.png', '11687.png', '14941.png', '13069.png', '12374.png', '23256.png', '22341.png', '06967.png', '12573.png', '11132.png', '22769.png', '18977.png', '00967.png', '23931.png', '15903.png', '08933.png', '06575.png', '19021.png', '11839.png', '08609.png', '08383.png', '10683.png', '00736.png', '09585.png', '08169.png', '04028.png', '13094.png', '21311.png', '04717.png', '24438.png', '22261.png', '08631.png', '23758.png', '00343.png', '09741.png', '02771.png', '20165.png', '18224.png', '04891.png', '18002.png', '21611.png', '07807.png', '07596.png', '17812.png', '24178.png', '00324.png', '01393.png', '11150.png', '05964.png', '24115.png', '24327.png', '13907.png', '12464.png', '17886.png', '20755.png', '16958.png', '12711.png', '02872.png', '03697.png', '13783.png', '13919.png', '04092.png', '17135.png', '06998.png', '15845.png', '09810.png', '05152.png', '21018.png', '20451.png', '01159.png', '22520.png', '10541.png', '14887.png', '05995.png', '10826.png', '04484.png', '01596.png', '09325.png', '24560.png', '18427.png', '10462.png', '15171.png', '15823.png', '03834.png', '23531.png', '02068.png', '14019.png', '14515.png', '20713.png', '20458.png', '24227.png', '11122.png', '07548.png', '14275.png', '03456.png', '15991.png', '06375.png', '20072.png', '17157.png', '23633.png', '17156.png', '08363.png', '04212.png', '01453.png', '20661.png', '12957.png', '17934.png', '13804.png', '14692.png', '11441.png', '17238.png', '18525.png', '00386.png', '02113.png', '22485.png', '16466.png', '08641.png', '21289.png', '24177.png', '04546.png', '14699.png', '01137.png', '00044.png', '03578.png', '12298.png', '05846.png', '22611.png', '11951.png', '08570.png', '13604.png', '13166.png', '08493.png', '20009.png', '07287.png', '11971.png', '09895.png', '22980.png', '11790.png', '07900.png', '03102.png', '13515.png', '00855.png', '11893.png', '07465.png', '05998.png', '22954.png', '21949.png', '17215.png', '01121.png', '03644.png', '05697.png', '21871.png', '03405.png', '02031.png', '23751.png', '13699.png', '10127.png', '21412.png', '16826.png', '09039.png', '23811.png', '21997.png', '20664.png', '04876.png', '04606.png', '01215.png', '24892.png', '17721.png', '24013.png', '09659.png', '15298.png', '01562.png', '17363.png', '02706.png', '23537.png', '14803.png', '15993.png', '10935.png', '23961.png', '17315.png', '06671.png', '21276.png', '07301.png', '11014.png', '03813.png', '10112.png', '00341.png', '20857.png', '21070.png', '15707.png', '15114.png', '14579.png', '08739.png', '13161.png', '05662.png', '21667.png', '12897.png', '11462.png', '21072.png', '20811.png', '16173.png', '02937.png', '04840.png', '08639.png', '18938.png', '03332.png', '18788.png', '22399.png', '06526.png', '04386.png', '21033.png', '19519.png', '08762.png', '04322.png', '13135.png', '01879.png', '24447.png', '15346.png', '10224.png', '17856.png', '06536.png', '14543.png', '22445.png', '01113.png', '14105.png', '21773.png', '10006.png', '01118.png', '04830.png', '07387.png', '00551.png', '08595.png', '14701.png', '00763.png', '02792.png', '22551.png', '02103.png', '16971.png', '04182.png', '12595.png', '08517.png', '09834.png', '06981.png', '22747.png', '09074.png', '11278.png', '08686.png', '04732.png', '17861.png', '13081.png', '08373.png', '13309.png', '09774.png', '13070.png', '23754.png', '01047.png', '10146.png', '18738.png', '12613.png', '16953.png', '12555.png', '18214.png', '02929.png', '16396.png', '11838.png', '14338.png', '24196.png', '10274.png', '04350.png', '11938.png', '08956.png', '05233.png', '13737.png', '07817.png', '03864.png', '04158.png', '04990.png', '04078.png', '15129.png', '13572.png', '09198.png', '21557.png', '06627.png', '00942.png', '02810.png', '00013.png', '16846.png', '03090.png', '12584.png', '06739.png', '20437.png', '18193.png', '19077.png', '20717.png', '19274.png', '01507.png', '03212.png', '01365.png', '06319.png', '01077.png', '23332.png', '04117.png', '22805.png', '19335.png', '12089.png', '12197.png', '22110.png', '10090.png', '04370.png', '10345.png', '19632.png', '14482.png', '03417.png', '15330.png', '00999.png', '24333.png', '17756.png', '14690.png', '22332.png', '17811.png', '11939.png', '12600.png', '03615.png', '00007.png', '14648.png', '02177.png', '12800.png', '08534.png', '16457.png', '08356.png', '03724.png', '08285.png', '22264.png', '05271.png', '21952.png', '01306.png', '04416.png', '24583.png', '08742.png', '16427.png', '23387.png', '07311.png', '14383.png', '15016.png', '15434.png', '11088.png', '12480.png', '02010.png', '11064.png', '16510.png', '07315.png', '16251.png', '09698.png', '15448.png', '23925.png', '22278.png', '08604.png', '01093.png', '23011.png', '02745.png', '23692.png', '09432.png', '22283.png', '00884.png', '14093.png', '06860.png', '17763.png', '06183.png', '19483.png', '07261.png', '23096.png', '07317.png', '05681.png', '09626.png', '03172.png', '20542.png', '14662.png', '01259.png', '18887.png', '10268.png', '16945.png', '04493.png', '12036.png', '00780.png', '08485.png', '04581.png', '04497.png', '14864.png', '09106.png', '24462.png', '20507.png', '17499.png', '07951.png', '12995.png', '06196.png', '22012.png', '08836.png', '04952.png', '19162.png', '22353.png', '07610.png', '18620.png', '05507.png', '00836.png', '18820.png', '21136.png', '10115.png', '23637.png', '00952.png', '08872.png', '04607.png', '08886.png', '02322.png', '06873.png', '09259.png', '05581.png', '04816.png', '04908.png', '12439.png', '00105.png', '03871.png', '07303.png', '07047.png', '14294.png', '09968.png', '09111.png', '22776.png', '02960.png', '14204.png', '21242.png', '18594.png', '20246.png', '06083.png', '22929.png', '15416.png', '03175.png', '02481.png', '07691.png', '24634.png', '24152.png', '13226.png', '15324.png', '02141.png', '22099.png', '02654.png', '13817.png', '03522.png', '03347.png', '03832.png', '02804.png', '14782.png', '06921.png', '00422.png', '21282.png', '01290.png', '10455.png', '09446.png', '07620.png', '09840.png', '02967.png', '05543.png', '13416.png', '05873.png', '12649.png', '23188.png', '17729.png', '14748.png', '09740.png', '24244.png', '04851.png', '04342.png', '05787.png', '17779.png', '08352.png', '13237.png', '08513.png', '22754.png', '11414.png', '21567.png', '02268.png', '11917.png', '19825.png', '04305.png', '18044.png', '02524.png', '24050.png', '22806.png', '00129.png', '13513.png', '00126.png', '14053.png', '23601.png', '20351.png', '23651.png', '15157.png', '14213.png', '07475.png', '00874.png', '24241.png', '13043.png', '23820.png', '08125.png', '24535.png', '16218.png', '02732.png', '08995.png', '12244.png', '14948.png', '00202.png', '01898.png', '07403.png', '17529.png', '14300.png', '07147.png', '18632.png', '20026.png', '06323.png', '03777.png', '05696.png', '22718.png', '10626.png', '11182.png', '03254.png', '19479.png', '15886.png', '12904.png', '07505.png', '01238.png', '22245.png', '02787.png', '11834.png', '03059.png', '16905.png', '21535.png', '22193.png', '19280.png', '13678.png', '12561.png', '22215.png', '10507.png', '03974.png', '07591.png', '11407.png', '24071.png', '21089.png', '12022.png', '24777.png', '13481.png', '00647.png', '09639.png', '07772.png', '03859.png', '24021.png', '16102.png', '19781.png', '18660.png', '05960.png', '10280.png', '17704.png', '02285.png', '22223.png', '20989.png', '22524.png', '00624.png', '16535.png', '11147.png', '16721.png', '16269.png', '24214.png', '04866.png', '24372.png', '01278.png', '06905.png', '09331.png', '03277.png', '00498.png', '07127.png', '03012.png', '19917.png', '15351.png', '04467.png', '24532.png', '15321.png', '16127.png', '06841.png', '20751.png', '08211.png', '23561.png', '20781.png', '18827.png', '23425.png', '04106.png', '10949.png', '23255.png', '03963.png', '23322.png', '20057.png', '15954.png', '04524.png', '03601.png', '19409.png', '18449.png', '14646.png', '20337.png', '10746.png', '02734.png', '19101.png', '14165.png', '08122.png', '13535.png', '20463.png', '02942.png', '19738.png', '24102.png', '00240.png', '06550.png', '02815.png', '19353.png', '07923.png', '13228.png', '13976.png', '19828.png', '01328.png', '12163.png', '12387.png', '16571.png', '12662.png', '18897.png', '14538.png', '24233.png', '01518.png', '05197.png', '17239.png', '07227.png', '02059.png', '09983.png', '07376.png', '18318.png', '13207.png', '05446.png', '09422.png', '06946.png', '12580.png', '07663.png', '10621.png', '20235.png', '19521.png', '23959.png', '06365.png', '01656.png', '11486.png', '24045.png', '22357.png', '17620.png', '18890.png', '17285.png', '01207.png', '11376.png', '17355.png', '19103.png', '17348.png', '08133.png', '16005.png', '10671.png', '13026.png', '24940.png', '03911.png', '04435.png', '01314.png', '04064.png', '08178.png', '08050.png', '04721.png', '24511.png', '12361.png', '22334.png', '07199.png', '21394.png', '08785.png', '16856.png', '16081.png', '10113.png', '12329.png', '07934.png', '17195.png', '04963.png', '12028.png', '14171.png', '07511.png', '22382.png', '12179.png', '20890.png', '04502.png', '04412.png', '00272.png', '20397.png', '08677.png', '07064.png', '07467.png', '24703.png', '09728.png', '20150.png', '18648.png', '24429.png', '09687.png', '22173.png', '12261.png', '05668.png', '06866.png', '03507.png', '08441.png', '11043.png', '19318.png', '03555.png', '15994.png', '13084.png', '20982.png', '21144.png', '08201.png', '14923.png', '09794.png', '17777.png', '02996.png', '03267.png', '13389.png', '04127.png', '22411.png', '20868.png', '13118.png', '03095.png', '00805.png', '22825.png', '22826.png', '03917.png', '09293.png', '19747.png', '16242.png', '20030.png', '00903.png', '24465.png', '15996.png', '13089.png', '06893.png', '22749.png', '16870.png', '18082.png', '23544.png', '11206.png', '24851.png', '11380.png', '22192.png', '02850.png', '06038.png', '09157.png', '11656.png', '02776.png', '03076.png', '12681.png', '16298.png', '20130.png', '00547.png', '07078.png', '00956.png', '07848.png', '11714.png', '00347.png', '06912.png', '20810.png', '24122.png', '10058.png', '04793.png', '15293.png', '15759.png', '17480.png', '01239.png', '16809.png', '00730.png', '20662.png', '10461.png', '13112.png', '07821.png', '08776.png', '09273.png', '22546.png', '01938.png', '05018.png', '20093.png', '05437.png', '23454.png', '05604.png', '09943.png', '04966.png', '02805.png', '08751.png', '16150.png']
    
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
            if epochs_since_start == 2:
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
        if epochs_since_start == 2:
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

        if i_iter == 56505:
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
