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
    a = ['24722.png', '13908.png', '03197.png', '01105.png', '22965.png', '08329.png', '11119.png', '15681.png', '13071.png', '00100.png', '03822.png', '04648.png', '24056.png', '20743.png', '10115.png', '12780.png', '12585.png', '08871.png', '17736.png', '19467.png', '01955.png', '20424.png', '12962.png', '13471.png', '11426.png', '21266.png', '21397.png', '16651.png', '15461.png', '16291.png', '20165.png', '14163.png', '08481.png', '22627.png', '02766.png', '20982.png', '17004.png', '17519.png', '23384.png', '10579.png', '24585.png', '20205.png', '14753.png', '04944.png', '18155.png', '24119.png', '10747.png', '20546.png', '01785.png', '18826.png', '08935.png', '24271.png', '18612.png', '10404.png', '19081.png', '24322.png', '21461.png', '02651.png', '12975.png', '13379.png', '13377.png', '17430.png', '12369.png', '01745.png', '08674.png', '20805.png', '15320.png', '16381.png', '16756.png', '23203.png', '01893.png', '04798.png', '00983.png', '10070.png', '22677.png', '03330.png', '13964.png', '14125.png', '17258.png', '18017.png', '08494.png', '08759.png', '01031.png', '08973.png', '08723.png', '16890.png', '02994.png', '20192.png', '12224.png', '20331.png', '19545.png', '02990.png', '17306.png', '04267.png', '24388.png', '13074.png', '05581.png', '19820.png', '08094.png', '23557.png', '01410.png', '16175.png', '23054.png', '13475.png', '06498.png', '21505.png', '11624.png', '22098.png', '19674.png', '19890.png', '12336.png', '22083.png', '03685.png', '20979.png', '02779.png', '14248.png', '24304.png', '01432.png', '08241.png', '23911.png', '11344.png', '16166.png', '08034.png', '10081.png', '22845.png', '17456.png', '21485.png', '18348.png', '03370.png', '04363.png', '17527.png', '11610.png', '06700.png', '00437.png', '06078.png', '06542.png', '14719.png', '24816.png', '16395.png', '06806.png', '16380.png', '02369.png', '17555.png', '03363.png', '13618.png', '05606.png', '01510.png', '15282.png', '07643.png', '09637.png', '01365.png', '13534.png', '09183.png', '05857.png', '01468.png', '08394.png', '08111.png', '23496.png', '10556.png', '09354.png', '20318.png', '14215.png', '08989.png', '02100.png', '15883.png', '01600.png', '15811.png', '08447.png', '21245.png', '02647.png', '01108.png', '00934.png', '01661.png', '17400.png', '03559.png', '12761.png', '20994.png', '24194.png', '13881.png', '23797.png', '03301.png', '09734.png', '14327.png', '17074.png', '00285.png', '10287.png', '16864.png', '21001.png', '10566.png', '19333.png', '12512.png', '11360.png', '21651.png', '06587.png', '04604.png', '20479.png', '23247.png', '22489.png', '11276.png', '16993.png', '17066.png', '24377.png', '17651.png', '19922.png', '11642.png', '17182.png', '12055.png', '16748.png', '16602.png', '22667.png', '18762.png', '08604.png', '19805.png', '04385.png', '17506.png', '07894.png', '18674.png', '12590.png', '04596.png', '02112.png', '23832.png', '19360.png', '04453.png', '10274.png', '19331.png', '07255.png', '15081.png', '19928.png', '24491.png', '21713.png', '00176.png', '19656.png', '02065.png', '22423.png', '21314.png', '20199.png', '23097.png', '14587.png', '11354.png', '14734.png', '24390.png', '04615.png', '06738.png', '18683.png', '02963.png', '24121.png', '08856.png', '24366.png', '12954.png', '11390.png', '03297.png', '03038.png', '21768.png', '09298.png', '02604.png', '06681.png', '15780.png', '21675.png', '01832.png', '12405.png', '20036.png', '19737.png', '13400.png', '12443.png', '13450.png', '20628.png', '10831.png', '15950.png', '01709.png', '06288.png', '17467.png', '18201.png', '24652.png', '05946.png', '18600.png', '03414.png', '16919.png', '18069.png', '03529.png', '13590.png', '15283.png', '24942.png', '17107.png', '23636.png', '16934.png', '05557.png', '03938.png', '24717.png', '24697.png', '02932.png', '05006.png', '02520.png', '20709.png', '20028.png', '13386.png', '13487.png', '14376.png', '15022.png', '14672.png', '13687.png', '15428.png', '21152.png', '06857.png', '22650.png', '24081.png', '15247.png', '20942.png', '08487.png', '13255.png', '00848.png', '02118.png', '14322.png', '21287.png', '16884.png', '02036.png', '15427.png', '00961.png', '05157.png', '24735.png', '23011.png', '10146.png', '12162.png', '17137.png', '21612.png', '14721.png', '04307.png', '09547.png', '10551.png', '10657.png', '17391.png', '07540.png', '07461.png', '10241.png', '18280.png', '17000.png', '10755.png', '10432.png', '20444.png', '04605.png', '15731.png', '00287.png', '11137.png', '23936.png', '20356.png', '00877.png', '19734.png', '18658.png', '10652.png', '15578.png', '05677.png', '16638.png', '08051.png', '11462.png', '24943.png', '21139.png', '03791.png', '07350.png', '01256.png', '14727.png', '14101.png', '23575.png', '13058.png', '10990.png', '11879.png', '20309.png', '09714.png', '23566.png', '16770.png', '02498.png', '06032.png', '23279.png', '18427.png', '09174.png', '06419.png', '08878.png', '23135.png', '17865.png', '19202.png', '03364.png', '15037.png', '22128.png', '12178.png', '19472.png', '08480.png', '22681.png', '00565.png', '04587.png', '11120.png', '07544.png', '16945.png', '17559.png', '09098.png', '00884.png', '09904.png', '11317.png', '22905.png', '00054.png', '18970.png', '04885.png', '10351.png', '05816.png', '10179.png', '07097.png', '23030.png', '08244.png', '19313.png', '03820.png', '02864.png', '06947.png', '18112.png', '19160.png', '14288.png', '16218.png', '24738.png', '04195.png', '11330.png', '13604.png', '02788.png', '19581.png', '07059.png', '10490.png', '19645.png', '20480.png', '17324.png', '20577.png', '21878.png', '20782.png', '22660.png', '02304.png', '00407.png', '00123.png', '05489.png', '14487.png', '10838.png', '09368.png', '09503.png', '05341.png', '16376.png', '20681.png', '15785.png', '13197.png', '05955.png', '23406.png', '07182.png', '19972.png', '16495.png', '01793.png', '03679.png', '00743.png', '11888.png', '18898.png', '22351.png', '23750.png', '19399.png', '05845.png', '14868.png', '21634.png', '06534.png', '06811.png', '18932.png', '06087.png', '16506.png', '13272.png', '04863.png', '00066.png', '19231.png', '02385.png', '18495.png', '21228.png', '07653.png', '04790.png', '04758.png', '22888.png', '00656.png', '00857.png', '20734.png', '06266.png', '06146.png', '12397.png', '01397.png', '23449.png', '07120.png', '17451.png', '13901.png', '24928.png', '03709.png', '24957.png', '14634.png', '20515.png', '21495.png', '19484.png', '21934.png', '23494.png', '01259.png', '06172.png', '06263.png', '12528.png', '05939.png', '00414.png', '01207.png', '10668.png', '17379.png', '04732.png', '09920.png', '16621.png', '17697.png', '14556.png', '10444.png', '16089.png', '14703.png', '14445.png', '04694.png', '06802.png', '05766.png', '14520.png', '12795.png', '12536.png', '15712.png', '18454.png', '22804.png', '15896.png', '08077.png', '04125.png', '22663.png', '00524.png', '01046.png', '05788.png', '05004.png', '16000.png', '03593.png', '02213.png', '06340.png', '08744.png', '20496.png', '24320.png', '17496.png', '06311.png', '17051.png', '01831.png', '21526.png', '19317.png', '04428.png', '20457.png', '01390.png', '10502.png', '01306.png', '12372.png', '20225.png', '03652.png', '14359.png', '05191.png', '24025.png', '12574.png', '20606.png', '13451.png', '08264.png', '13470.png', '23190.png', '14193.png', '23132.png', '05646.png', '24452.png', '19196.png', '02929.png', '15057.png', '08594.png', '02509.png', '10105.png', '22680.png', '14083.png', '23878.png', '02250.png', '16400.png', '22046.png', '11817.png', '07908.png', '22302.png', '01294.png', '21128.png', '13876.png', '23845.png', '10787.png', '19130.png', '00908.png', '22596.png', '13203.png', '07927.png', '18012.png', '15249.png', '17972.png', '23776.png', '14673.png', '19728.png', '22436.png', '22444.png', '17208.png', '16877.png', '18543.png', '05666.png', '18476.png', '00739.png', '20568.png', '19211.png', '01287.png', '21900.png', '24269.png', '05780.png', '13580.png', '14373.png', '14988.png', '06939.png', '23568.png', '24528.png', '08942.png', '19688.png', '07027.png', '07666.png', '11142.png', '10857.png', '17063.png', '16471.png', '03733.png', '12310.png', '11020.png', '09395.png', '03053.png', '15068.png', '20831.png', '00613.png', '14518.png', '17279.png', '17649.png', '16378.png', '21912.png', '02972.png', '00620.png', '04891.png', '06513.png', '05706.png', '10607.png', '16375.png', '20653.png', '08130.png', '01658.png', '04508.png', '06553.png', '05818.png', '20471.png', '08175.png', '11231.png', '06580.png', '08958.png', '24577.png', '17471.png', '18431.png', '17539.png', '24464.png', '00899.png', '04502.png', '05608.png', '06441.png', '07515.png', '20322.png', '20508.png', '12447.png', '05338.png', '12050.png', '22483.png', '11124.png', '15739.png', '09392.png', '11250.png', '04877.png', '14964.png', '02702.png', '07067.png', '15246.png', '04782.png', '16959.png', '05836.png', '03961.png', '14561.png', '07724.png', '14497.png', '11323.png', '13208.png', '06443.png', '01712.png', '08632.png', '18895.png', '24082.png', '02847.png', '18994.png', '22728.png', '12064.png', '01239.png', '07845.png', '22734.png', '21529.png', '16483.png', '18991.png', '07866.png', '03876.png', '00949.png', '21105.png', '12674.png', '06481.png', '19115.png', '15640.png', '06502.png', '21956.png', '11819.png', '03495.png', '21564.png', '16948.png', '03419.png', '10297.png', '09286.png', '06510.png', '12177.png', '02094.png', '16310.png', '14003.png', '04531.png', '15668.png', '06842.png', '00509.png', '13569.png', '17326.png', '05156.png', '10169.png', '11782.png', '12919.png', '01710.png', '24339.png', '04933.png', '06143.png', '16711.png', '14021.png', '02854.png', '08476.png', '13723.png', '15672.png', '15107.png', '07838.png', '15624.png', '08378.png', '07961.png', '19184.png', '21938.png', '05308.png', '02863.png', '17878.png', '05571.png', '04898.png', '12141.png', '09430.png', '03055.png', '01024.png', '08889.png', '01874.png', '07576.png', '04838.png', '19171.png', '11362.png', '18515.png', '03843.png', '00578.png', '12390.png', '16084.png', '13140.png', '00663.png', '08095.png', '02952.png', '18501.png', '06468.png', '22471.png', '02328.png', '24960.png', '21189.png', '15463.png', '20134.png', '17611.png', '07284.png', '05521.png', '20448.png', '13562.png', '10198.png', '04867.png', '10633.png', '03206.png', '14669.png', '13679.png', '18106.png', '23513.png', '21072.png', '10904.png', '12773.png', '02885.png', '03918.png', '14828.png', '04714.png', '07262.png', '07740.png', '03632.png', '14642.png', '24167.png', '21837.png', '02265.png', '12273.png', '06744.png', '04954.png', '09736.png', '03108.png', '12842.png', '16464.png', '02583.png', '20194.png', '12915.png', '19935.png', '22391.png', '16238.png', '13467.png', '03393.png', '07164.png', '15772.png', '20005.png', '23424.png', '00246.png', '24915.png', '22020.png', '16261.png', '24148.png', '04338.png', '14823.png', '17671.png', '07884.png', '12009.png', '12148.png', '06493.png', '18425.png', '12522.png', '06195.png', '18535.png', '22186.png', '16326.png', '06906.png', '02269.png', '02048.png', '18701.png', '00515.png', '03537.png', '16252.png', '22651.png', '05607.png', '21319.png', '06500.png', '11122.png', '04577.png', '20758.png', '11504.png', '22919.png', '07297.png', '07856.png', '02025.png', '21583.png', '13543.png', '15507.png', '17742.png', '05137.png', '00803.png', '17172.png', '05926.png', '22969.png', '13906.png', '01377.png', '05134.png', '16042.png', '00699.png', '20812.png', '00726.png', '16676.png', '01511.png', '09483.png', '12614.png', '08499.png', '19576.png', '01213.png', '08905.png', '23653.png', '17190.png', '24258.png', '00018.png', '08386.png', '17593.png', '03420.png', '00227.png', '21179.png', '18627.png', '10005.png', '14575.png', '19914.png', '09767.png', '18866.png', '16565.png', '17979.png', '14578.png', '12081.png', '03963.png', '24819.png', '17667.png', '02630.png', '11516.png', '11059.png', '16267.png', '20555.png', '22892.png', '08336.png', '20482.png', '08887.png', '11766.png', '16130.png', '03864.png', '01917.png', '05289.png', '09690.png', '00118.png', '14266.png', '16215.png', '07096.png', '20252.png', '12738.png', '17312.png', '18282.png', '16851.png', '19700.png', '15885.png', '00984.png', '14758.png', '19855.png', '11365.png', '06840.png', '23505.png', '04138.png', '12022.png', '16584.png', '18757.png', '18207.png', '19490.png', '21711.png', '20466.png', '23295.png', '05545.png', '16403.png', '10473.png', '16462.png', '21840.png', '02179.png', '12717.png', '24799.png', '06503.png', '00865.png', '14179.png', '09829.png', '19594.png', '02466.png', '22822.png', '23865.png', '13241.png', '06115.png', '00725.png', '24739.png', '18100.png', '21646.png', '12765.png', '15603.png', '04066.png', '15209.png', '15949.png', '24079.png', '14682.png', '04233.png', '01281.png', '16420.png', '06289.png', '02684.png', '04107.png', '01459.png', '17599.png', '17436.png', '05976.png', '21533.png', '16398.png', '23753.png', '10896.png', '11063.png', '12556.png', '09289.png', '09534.png', '06505.png', '17468.png', '00463.png', '09844.png', '09634.png', '06833.png', '22567.png', '06358.png', '03854.png', '10240.png', '24072.png', '04729.png', '21916.png', '04955.png', '16391.png', '09647.png', '17385.png', '12436.png', '19509.png', '03040.png', '09894.png', '06785.png', '21273.png', '22857.png', '20334.png', '03341.png', '00744.png', '00926.png', '19440.png', '15211.png', '14460.png', '10421.png', '02551.png', '01093.png', '08994.png', '22749.png', '12502.png', '00159.png', '12626.png', '11044.png', '18065.png', '22582.png', '11027.png', '14981.png', '09757.png', '20392.png', '02732.png', '03486.png', '11762.png', '21775.png', '02479.png', '02741.png', '21259.png', '00200.png', '21045.png', '02348.png', '07103.png', '21137.png', '04602.png', '15395.png', '08192.png', '19761.png', '22085.png', '13027.png', '21415.png', '09275.png', '16122.png', '18582.png', '16905.png', '17500.png', '16415.png', '22911.png', '01595.png', '16809.png', '20427.png', '18952.png', '17647.png', '04505.png', '10765.png', '10717.png', '20246.png', '02204.png', '06307.png', '21238.png', '03065.png', '14508.png', '00836.png', '24642.png', '20349.png', '01629.png', '08050.png', '24151.png', '09036.png', '22983.png', '03796.png', '21882.png', '15236.png', '10768.png', '17689.png', '01662.png', '16726.png', '15315.png', '18513.png', '04660.png', '18294.png', '08888.png', '19084.png', '23401.png', '23726.png', '21601.png', '20956.png', '03606.png', '24421.png', '07815.png', '19197.png', '03948.png', '21749.png', '24063.png', '06341.png', '10309.png', '00166.png', '12166.png', '20498.png', '04252.png', '20347.png', '21101.png', '15146.png', '20608.png', '09415.png', '19020.png', '18291.png', '20896.png', '06757.png', '15023.png', '14414.png', '20627.png', '03323.png', '15823.png', '15918.png', '03893.png', '03624.png', '23671.png', '24172.png', '16921.png', '20796.png', '19054.png', '15554.png', '16822.png', '24511.png', '05875.png', '23893.png', '15104.png', '00827.png', '05442.png', '18469.png', '18240.png', '12726.png', '21527.png', '06621.png', '18137.png', '11882.png', '20208.png', '18801.png', '07415.png', '05441.png', '07501.png', '02408.png', '15988.png', '21570.png', '17349.png', '13236.png', '06313.png', '07212.png', '22180.png', '22585.png', '15972.png', '06547.png', '04181.png', '12266.png', '18486.png', '11405.png', '20716.png', '22639.png', '22643.png', '12098.png', '10004.png', '04289.png', '20561.png', '14362.png', '21021.png', '17586.png', '17020.png', '05324.png', '00719.png', '08412.png', '09299.png', '16964.png', '02086.png', '16307.png', '13315.png', '02674.png', '03067.png', '00508.png', '01013.png', '06854.png', '21635.png', '21561.png', '14599.png', '11270.png', '15470.png', '04563.png', '08616.png', '18988.png', '08893.png', '24334.png', '24672.png', '19389.png', '11933.png', '15142.png', '01692.png', '14890.png', '00564.png', '21921.png', '15351.png', '19042.png', '24715.png', '11224.png', '18815.png', '21194.png', '09237.png', '12351.png', '19349.png', '12085.png', '20071.png', '16384.png', '07473.png', '01216.png', '04918.png', '12283.png', '14016.png', '21394.png', '23630.png', '22656.png', '04971.png', '24859.png', '00655.png', '09853.png', '14572.png', '20687.png', '20472.png', '05640.png', '23963.png', '05365.png', '23563.png', '09850.png', '16853.png', '03496.png', '16750.png', '24033.png', '12560.png', '12941.png', '20682.png', '12474.png', '21249.png', '17120.png', '09397.png', '19640.png', '12126.png', '21597.png', '14899.png', '02700.png', '01471.png', '01617.png', '22529.png', '24474.png', '03725.png', '21366.png', '03677.png', '18413.png', '08115.png', '04688.png', '19686.png', '19896.png', '09640.png', '21639.png', '03212.png', '04105.png', '13736.png', '24171.png', '18393.png', '09351.png', '20526.png', '08432.png', '16966.png', '19930.png', '16786.png', '06184.png', '14780.png', '22452.png', '01908.png', '05965.png', '05445.png', '09077.png', '10715.png', '02295.png', '04981.png', '22345.png', '17261.png', '23208.png', '15853.png', '04961.png', '07871.png', '04916.png', '22914.png', '16240.png', '17792.png', '01135.png', '19511.png', '10080.png', '11373.png', '16180.png', '23680.png', '11816.png', '11889.png', '12028.png', '12565.png', '07719.png', '18980.png', '00691.png', '11920.png', '03338.png']
    
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
