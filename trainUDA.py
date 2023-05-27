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
    a = ['18316.png', '09598.png', '13688.png', '21868.png', '02879.png', '08691.png', '20589.png', '04994.png', '08107.png', '18697.png', '15417.png', '08445.png', '18484.png', '06209.png', '23209.png', '04161.png', '04593.png', '20973.png', '14880.png', '19226.png', '13194.png', '04895.png', '17227.png', '06945.png', '08526.png', '12075.png', '20757.png', '01495.png', '18876.png', '09140.png', '12675.png', '21021.png', '20843.png', '12097.png', '06736.png', '13297.png', '01551.png', '02364.png', '11615.png', '08419.png', '24760.png', '18040.png', '22416.png', '01410.png', '14174.png', '00028.png', '23062.png', '15007.png', '06597.png', '11526.png', '14274.png', '05789.png', '05753.png', '10709.png', '16064.png', '05024.png', '02209.png', '17826.png', '00598.png', '19220.png', '09116.png', '11252.png', '02625.png', '10259.png', '15002.png', '10237.png', '03089.png', '15826.png', '14999.png', '08054.png', '10968.png', '01689.png', '00390.png', '10016.png', '12348.png', '04139.png', '03449.png', '21442.png', '01089.png', '09974.png', '08245.png', '21718.png', '21523.png', '22150.png', '18276.png', '14138.png', '11752.png', '22250.png', '16126.png', '18728.png', '21917.png', '23306.png', '01766.png', '19794.png', '04521.png', '11848.png', '17234.png', '18872.png', '16509.png', '17069.png', '10969.png', '11992.png', '06182.png', '18490.png', '11287.png', '04528.png', '05048.png', '04048.png', '02056.png', '01087.png', '17609.png', '04478.png', '09700.png', '23894.png', '11481.png', '16802.png', '24058.png', '19465.png', '23686.png', '24459.png', '04738.png', '01458.png', '20309.png', '07833.png', '10272.png', '22513.png', '15713.png', '14173.png', '24707.png', '02406.png', '09646.png', '02557.png', '12663.png', '16960.png', '24302.png', '05169.png', '17354.png', '10993.png', '08324.png', '18561.png', '10472.png', '08873.png', '21681.png', '02518.png', '21842.png', '04727.png', '07941.png', '07017.png', '19609.png', '15594.png', '13432.png', '04868.png', '08617.png', '04861.png', '16563.png', '14558.png', '24084.png', '19352.png', '02198.png', '02512.png', '01181.png', '06038.png', '07858.png', '17928.png', '11644.png', '19723.png', '00766.png', '06131.png', '09037.png', '07757.png', '10591.png', '15118.png', '13434.png', '19917.png', '24661.png', '23608.png', '15689.png', '24520.png', '14219.png', '18831.png', '04638.png', '13738.png', '00799.png', '23727.png', '23460.png', '10959.png', '03487.png', '16364.png', '24214.png', '06977.png', '03962.png', '04536.png', '04144.png', '03154.png', '09890.png', '03208.png', '03055.png', '18394.png', '24044.png', '09746.png', '09728.png', '04643.png', '10315.png', '01165.png', '16164.png', '08219.png', '15125.png', '01188.png', '06697.png', '18539.png', '12543.png', '21348.png', '16580.png', '20264.png', '01639.png', '04942.png', '09687.png', '05150.png', '15745.png', '21976.png', '10887.png', '03513.png', '07854.png', '09834.png', '01582.png', '14259.png', '01941.png', '16215.png', '10244.png', '21798.png', '04229.png', '09182.png', '03025.png', '09330.png', '21777.png', '02514.png', '03835.png', '16824.png', '17303.png', '23793.png', '12702.png', '04902.png', '11041.png', '22285.png', '16864.png', '19669.png', '05238.png', '20473.png', '20125.png', '24117.png', '12889.png', '18304.png', '03933.png', '10355.png', '18315.png', '21507.png', '18289.png', '18921.png', '13905.png', '13151.png', '09184.png', '05735.png', '12673.png', '24626.png', '16871.png', '24913.png', '06913.png', '05337.png', '17246.png', '09966.png', '17282.png', '00503.png', '24655.png', '22698.png', '07170.png', '02641.png', '00720.png', '06102.png', '20749.png', '09716.png', '19910.png', '18389.png', '09416.png', '12435.png', '05634.png', '08026.png', '01817.png', '24112.png', '21972.png', '19969.png', '01200.png', '01477.png', '00294.png', '10943.png', '05986.png', '10308.png', '00130.png', '19035.png', '14942.png', '21343.png', '12651.png', '02376.png', '21659.png', '23940.png', '00769.png', '23810.png', '03787.png', '22354.png', '15161.png', '18010.png', '01872.png', '05924.png', '07936.png', '11151.png', '01016.png', '16933.png', '18900.png', '09992.png', '12552.png', '15348.png', '21667.png', '16486.png', '08826.png', '17363.png', '17412.png', '21492.png', '13918.png', '07989.png', '04574.png', '01014.png', '03302.png', '04596.png', '06663.png', '05965.png', '14953.png', '17788.png', '06469.png', '24092.png', '18551.png', '07149.png', '20761.png', '12018.png', '13922.png', '02961.png', '05201.png', '01411.png', '18200.png', '09893.png', '17962.png', '08562.png', '03765.png', '00123.png', '19345.png', '13950.png', '20870.png', '21163.png', '06295.png', '24423.png', '01526.png', '03826.png', '18858.png', '14440.png', '24204.png', '14891.png', '12413.png', '13094.png', '04241.png', '13709.png', '23467.png', '12792.png', '12010.png', '05022.png', '21781.png', '20043.png', '00470.png', '12170.png', '21744.png', '12485.png', '20211.png', '14752.png', '23984.png', '03585.png', '16778.png', '15932.png', '00481.png', '14850.png', '06820.png', '10229.png', '24452.png', '07092.png', '09636.png', '08652.png', '03786.png', '11964.png', '14987.png', '01427.png', '11759.png', '02663.png', '16347.png', '10181.png', '04919.png', '14221.png', '20146.png', '03609.png', '02849.png', '07967.png', '15907.png', '21055.png', '03094.png', '14823.png', '04599.png', '07182.png', '12142.png', '08460.png', '08092.png', '02584.png', '21702.png', '06856.png', '12027.png', '23142.png', '00410.png', '14629.png', '12914.png', '12616.png', '03442.png', '22187.png', '20549.png', '03329.png', '21706.png', '22994.png', '12692.png', '11208.png', '03763.png', '19138.png', '08923.png', '00580.png', '13674.png', '21497.png', '21515.png', '02434.png', '14355.png', '15422.png', '20998.png', '11209.png', '19176.png', '15772.png', '13253.png', '00027.png', '06780.png', '15335.png', '22878.png', '05216.png', '19619.png', '00422.png', '08124.png', '22791.png', '05921.png', '04535.png', '02145.png', '19126.png', '11455.png', '09238.png', '20254.png', '14535.png', '19503.png', '20416.png', '11815.png', '17669.png', '23751.png', '10381.png', '03925.png', '00673.png', '19145.png', '11874.png', '04992.png', '21881.png', '16607.png', '23293.png', '15252.png', '19529.png', '00059.png', '20941.png', '20788.png', '03578.png', '13852.png', '08045.png', '08779.png', '22162.png', '15081.png', '20080.png', '13202.png', '07553.png', '21132.png', '05206.png', '06111.png', '04342.png', '17480.png', '14216.png', '23925.png', '07959.png', '16260.png', '16541.png', '05596.png', '20102.png', '12085.png', '09136.png', '19569.png', '12123.png', '10006.png', '15113.png', '24644.png', '05327.png', '07447.png', '20985.png', '05906.png', '16808.png', '01166.png', '20265.png', '19829.png', '21093.png', '06198.png', '19598.png', '04959.png', '13729.png', '23570.png', '07166.png', '24949.png', '18613.png', '02244.png', '10164.png', '03828.png', '16821.png', '15631.png', '01781.png', '02480.png', '04784.png', '15171.png', '24323.png', '07427.png', '24814.png', '15836.png', '10648.png', '13554.png', '11739.png', '04647.png', '19140.png', '00423.png', '23357.png', '07123.png', '03022.png', '24171.png', '06180.png', '14024.png', '18379.png', '17971.png', '04089.png', '09444.png', '06363.png', '19803.png', '19549.png', '11380.png', '22208.png', '10931.png', '03432.png', '17774.png', '08625.png', '04852.png', '22331.png', '20188.png', '11235.png', '23318.png', '24087.png', '20905.png', '00094.png', '18988.png', '19079.png', '10393.png', '01750.png', '22807.png', '16151.png', '04520.png', '13325.png', '03430.png', '16680.png', '20902.png', '11030.png', '01391.png', '12900.png', '15654.png', '12398.png', '07448.png', '06252.png', '04090.png', '23440.png', '15127.png', '21112.png', '02736.png', '08333.png', '00365.png', '24681.png', '01829.png', '15352.png', '00413.png', '03632.png', '02976.png', '09651.png', '11195.png', '08584.png', '04184.png', '02903.png', '12956.png', '21391.png', '01786.png', '01958.png', '03010.png', '03050.png', '11258.png', '24887.png', '05203.png', '22323.png', '13792.png', '18847.png', '00116.png', '06067.png', '22121.png', '01579.png', '19408.png', '16028.png', '23944.png', '05852.png', '18556.png', '10129.png', '15840.png', '22039.png', '03642.png', '23712.png', '16910.png', '07285.png', '00603.png', '17466.png', '02103.png', '06243.png', '23275.png', '00076.png', '18792.png', '10099.png', '18426.png', '08296.png', '04447.png', '24048.png', '15129.png', '18815.png', '19369.png', '08471.png', '05670.png', '02454.png', '03233.png', '23212.png', '18293.png', '22356.png', '07145.png', '24584.png', '10280.png', '05729.png', '01568.png', '19268.png', '24462.png', '22335.png', '19606.png', '15739.png', '10287.png', '13548.png', '00542.png', '04333.png', '04570.png', '01871.png', '13533.png', '15079.png', '13998.png', '06716.png', '07632.png', '13212.png', '24069.png', '08207.png', '09823.png', '21246.png', '01001.png', '18782.png', '00042.png', '23382.png', '05846.png', '22949.png', '12045.png', '19555.png', '18136.png', '00741.png', '06027.png', '10626.png', '19002.png', '13343.png', '07316.png', '07579.png', '00901.png', '22607.png', '12907.png', '02066.png', '04033.png', '17149.png', '22522.png', '04252.png', '04460.png', '06903.png', '17520.png', '07907.png', '04019.png', '05359.png', '21597.png', '00278.png', '01366.png', '17168.png', '24573.png', '12347.png', '09717.png', '08095.png', '19339.png', '11302.png', '07141.png', '19191.png', '21404.png', '02215.png', '17627.png', '07394.png', '18859.png', '14047.png', '04825.png', '23573.png', '20903.png', '02580.png', '16903.png', '14781.png', '03255.png', '04374.png', '07681.png', '03530.png', '05295.png', '10185.png', '08402.png', '14529.png', '14091.png', '01581.png', '21038.png', '00438.png', '16219.png', '17415.png', '23603.png', '08085.png', '00648.png', '18234.png', '20797.png', '19385.png', '16433.png', '01678.png', '24478.png', '20027.png', '18884.png', '11449.png', '01640.png', '12539.png', '14293.png', '15714.png', '18931.png', '15774.png', '19276.png', '04152.png', '19560.png', '11573.png', '22298.png', '13220.png', '09209.png', '00841.png', '15038.png', '03227.png', '19447.png', '15705.png', '23678.png', '21719.png', '21872.png', '18025.png', '06989.png', '05734.png', '05055.png', '23077.png', '00134.png', '07419.png', '23824.png', '14868.png', '01815.png', '08279.png', '00087.png', '08454.png', '18346.png', '01238.png', '15320.png', '03238.png', '05214.png', '24910.png', '17239.png', '00564.png', '07698.png', '22565.png', '12689.png', '01372.png', '17570.png', '06998.png', '15049.png', '21595.png', '03557.png', '17467.png', '00276.png', '16730.png', '07837.png', '13136.png', '24103.png', '16031.png', '10194.png', '24410.png', '07449.png', '17359.png', '24393.png', '15180.png', '09295.png', '15811.png', '18992.png', '20077.png', '06479.png', '06333.png', '23909.png', '21567.png', '06643.png', '20981.png', '01950.png', '08184.png', '04837.png', '09057.png', '19407.png', '24585.png', '23931.png', '14722.png', '10358.png', '06556.png', '09502.png', '18877.png', '15327.png', '20532.png', '12625.png', '05064.png', '03274.png', '02707.png', '15236.png', '01723.png', '16587.png', '19649.png', '06311.png', '24572.png', '19916.png', '21802.png', '09680.png', '13744.png', '18874.png', '21184.png', '10352.png', '24438.png', '09283.png', '13854.png', '02759.png', '11671.png', '16797.png', '01621.png', '09176.png', '03852.png', '09800.png', '02679.png', '10300.png', '10646.png', '18363.png', '06402.png', '07183.png', '00300.png', '04953.png', '03875.png', '24120.png', '21887.png', '14243.png', '09103.png', '18486.png', '10875.png', '08090.png', '05099.png', '10823.png', '03896.png', '07329.png', '03968.png', '08290.png', '23054.png', '15003.png', '20558.png', '05492.png', '06952.png', '22757.png', '21079.png', '16306.png', '10707.png', '18477.png', '01072.png', '15128.png', '00641.png', '09452.png', '15212.png', '10868.png', '01666.png', '16167.png', '14824.png', '17187.png', '12168.png', '07369.png', '04777.png', '08467.png', '18537.png', '12581.png', '06335.png', '20527.png', '16366.png', '08131.png', '10794.png', '01874.png', '05663.png', '13935.png', '05360.png', '18996.png', '04182.png', '17721.png', '05856.png', '20631.png', '00520.png', '20108.png', '11595.png', '20681.png', '03573.png', '04660.png', '04752.png', '19843.png', '07119.png', '05777.png', '10033.png', '15160.png', '00845.png', '06275.png', '04791.png', '10681.png', '01128.png', '20032.png', '22365.png', '14115.png', '01143.png', '11923.png', '16828.png', '07089.png', '08867.png', '03337.png', '03676.png', '07497.png', '21927.png', '16743.png', '17945.png', '02073.png', '16335.png', '12776.png', '09002.png', '19816.png', '14979.png', '03595.png', '16054.png', '16971.png', '17560.png', '16065.png', '03443.png', '17683.png', '10224.png', '07098.png', '18280.png', '22453.png', '01155.png', '05981.png', '01261.png', '10313.png', '08497.png', '04947.png', '06870.png', '08636.png', '20412.png', '18734.png', '01443.png', '15759.png', '23332.png', '18343.png', '10356.png', '14516.png', '23845.png', '06918.png', '15108.png', '21510.png', '12248.png', '02097.png', '06021.png', '16220.png', '03955.png', '00924.png', '15110.png', '18078.png', '19536.png', '17573.png', '01696.png', '19941.png', '12644.png', '18554.png', '12830.png', '12898.png', '16363.png', '14612.png', '15196.png', '15412.png', '24138.png', '08022.png', '11630.png', '14132.png', '14130.png', '05247.png', '05124.png', '13233.png', '11426.png', '10650.png', '12461.png', '12997.png', '12414.png', '18565.png', '21851.png', '15048.png', '19895.png', '09451.png', '03537.png', '08567.png', '18624.png', '09674.png', '06463.png', '22045.png', '16608.png', '17080.png', '22191.png', '04650.png', '16301.png', '04356.png', '08392.png', '06598.png', '13766.png', '11577.png', '16543.png', '01322.png', '15248.png', '00162.png', '02129.png', '17365.png', '21288.png', '09623.png', '09054.png', '13221.png', '09397.png', '13207.png', '02410.png', '11059.png', '12386.png', '14380.png', '12088.png', '22267.png', '02773.png', '18649.png', '17248.png', '00141.png', '00261.png', '08323.png', '14178.png', '18146.png', '13742.png', '20682.png', '17181.png', '10085.png', '17036.png', '02326.png', '24722.png', '24764.png', '00985.png', '00685.png', '13395.png', '17801.png', '18453.png', '04462.png', '13051.png', '17111.png', '14528.png', '11097.png', '00373.png', '00788.png', '04245.png', '13428.png', '06819.png', '06221.png', '13929.png', '17160.png', '17729.png', '04327.png', '21013.png', '24406.png', '00144.png', '10297.png', '23637.png', '07900.png', '08888.png', '24345.png', '03263.png', '21454.png', '23151.png', '18425.png', '03372.png', '19251.png', '06400.png', '20155.png', '23913.png', '07822.png', '08746.png', '11736.png', '09962.png', '10326.png', '07118.png', '02224.png', '16777.png', '23128.png', '16670.png', '06660.png', '20638.png', '10466.png', '12561.png', '06200.png', '24749.png', '12979.png', '14006.png', '21812.png', '23782.png', '18691.png', '01019.png', '06745.png', '07538.png', '20817.png', '04667.png', '02538.png', '08228.png', '13040.png', '03767.png', '09509.png', '04288.png', '18528.png', '09105.png', '06649.png', '00126.png', '20255.png', '14206.png', '09785.png', '14742.png', '16705.png', '14813.png', '16294.png', '06158.png', '20563.png', '24863.png', '13341.png', '11475.png', '20385.png', '07576.png', '23964.png', '04858.png', '20956.png', '12913.png', '19897.png', '17112.png', '07881.png', '13527.png', '22296.png', '18430.png', '24173.png', '14019.png', '01857.png', '04243.png', '10415.png', '00699.png', '00125.png', '21317.png', '24924.png', '14129.png', '09818.png', '05161.png', '06415.png', '15920.png', '06914.png', '17585.png', '15061.png', '20506.png', '15767.png', '18852.png', '02013.png', '02724.png', '02810.png', '11009.png', '13947.png', '21089.png', '11416.png', '12473.png', '19642.png', '00324.png', '10212.png', '09443.png', '02767.png', '02883.png', '15388.png', '20292.png', '03305.png', '02622.png', '21462.png', '17409.png', '09747.png', '15098.png', '22987.png', '10908.png', '08866.png', '13210.png', '12231.png', '08756.png', '08984.png', '04845.png', '20882.png', '08167.png', '24954.png', '08491.png', '10552.png', '22952.png', '16928.png', '14993.png', '06743.png', '21693.png', '12207.png', '01433.png', '17436.png', '12402.png', '17001.png', '08072.png', '20791.png', '08301.png', '17161.png', '04013.png', '13009.png', '23790.png', '15600.png', '21475.png', '19349.png', '15660.png', '24291.png', '18489.png', '14306.png', '17316.png', '20442.png', '16009.png', '05293.png', '22118.png', '06947.png', '11614.png', '06657.png', '03127.png', '18444.png', '11476.png', '05929.png', '24324.png', '15667.png', '02115.png', '07877.png', '06491.png', '09219.png', '11133.png', '01283.png', '00926.png', '17206.png', '10980.png', '00866.png', '05697.png', '07326.png', '10789.png', '16420.png', '10238.png', '11307.png', '23286.png', '10682.png', '21049.png', '09581.png', '09653.png', '21121.png', '17509.png', '07217.png', '23980.png', '08593.png', '19568.png', '13798.png', '22096.png', '16003.png', '20408.png', '19736.png', '24641.png', '17459.png', '04258.png', '06145.png', '19110.png', '23504.png', '16809.png', '12884.png', '01975.png', '14347.png', '19852.png', '06609.png', '16819.png', '07947.png', '13126.png', '02808.png', '07718.png', '16667.png', '20413.png', '09450.png', '04631.png', '24441.png', '11290.png', '04228.png', '00813.png', '03804.png', '00537.png', '14428.png', '15550.png', '00840.png', '20766.png', '18777.png', '07646.png', '00158.png', '17101.png', '22725.png', '04744.png', '14985.png', '08369.png', '01111.png', '02864.png', '05979.png', '04292.png', '03180.png', '00460.png', '08127.png', '08754.png', '15205.png', '21614.png', '03460.png', '15308.png', '02523.png', '00389.png', '07330.png', '00177.png', '04686.png', '13699.png', '16562.png', '12860.png', '11499.png', '14043.png', '09044.png', '23644.png', '10769.png', '04968.png', '09326.png', '18885.png', '04860.png', '21282.png', '13258.png', '14605.png', '20685.png', '14330.png', '09198.png', '16101.png', '16865.png', '18022.png', '00490.png', '02166.png', '04251.png', '04108.png', '15910.png', '00540.png', '02815.png', '09080.png', '04304.png', '16262.png', '04059.png', '09280.png', '22224.png', '07780.png', '02012.png', '00149.png', '15864.png', '02003.png', '17842.png', '09669.png', '04835.png', '01576.png', '16750.png', '21555.png', '24319.png', '20538.png', '11632.png', '16423.png', '01756.png', '22581.png', '19201.png', '11218.png', '07895.png', '22294.png', '09832.png', '02542.png', '06885.png', '18355.png', '12481.png', '02919.png', '08438.png', '18241.png', '15116.png', '04313.png', '04905.png', '16291.png', '19815.png', '02658.png', '19594.png', '14214.png', '18990.png', '21765.png', '07129.png', '15495.png', '12118.png', '07904.png', '01532.png', '03790.png', '00591.png', '19884.png', '16139.png', '03604.png', '06482.png', '19925.png', '06985.png', '11834.png', '12357.png', '06899.png', '20574.png', '11892.png', '11961.png', '12429.png', '00398.png', '16357.png', '08760.png', '08742.png', '05443.png', '22660.png', '23117.png', '13769.png', '06706.png', '12791.png', '13465.png', '20513.png', '01713.png', '16897.png', '03332.png', '12554.png', '17099.png', '04965.png', '22696.png', '18515.png', '17914.png', '12262.png', '18727.png', '06665.png', '20249.png', '11512.png', '14392.png', '23250.png', '18122.png', '11185.png', '07361.png', '12120.png', '10008.png', '05639.png', '01973.png', '20117.png', '03128.png', '11979.png', '20220.png', '10895.png', '19270.png', '22662.png', '23416.png', '21048.png', '11587.png', '10898.png', '01844.png', '17778.png', '08694.png', '22590.png', '20084.png', '09662.png', '10256.png', '21803.png', '16278.png', '09492.png', '10458.png', '20055.png', '22272.png', '00959.png', '21517.png', '05103.png', '20287.png', '02809.png', '01385.png', '15273.png', '13426.png', '09222.png', '23583.png', '02713.png', '10179.png', '17261.png', '00747.png', '16042.png', '07522.png', '12687.png', '14410.png', '21669.png', '08049.png', '19914.png', '00651.png', '22237.png', '19675.png', '02424.png', '08778.png', '07546.png', '05990.png', '18396.png', '08098.png', '08205.png', '19802.png', '24864.png', '19474.png', '13832.png', '16456.png', '09083.png', '04428.png', '09495.png', '06799.png', '14910.png', '15698.png', '15329.png', '04576.png', '13090.png', '24879.png', '11182.png', '04205.png', '11866.png', '21003.png', '02044.png', '14279.png', '08985.png', '14223.png', '00403.png', '13303.png', '23649.png', '03389.png', '20234.png', '12300.png', '10271.png', '13595.png', '15059.png', '03576.png', '07082.png', '17794.png', '09597.png', '23013.png', '17120.png', '19334.png', '17714.png', '01329.png', '08001.png', '12953.png', '24239.png', '22003.png', '02460.png', '04501.png', '03117.png', '20911.png', '08280.png', '16688.png', '22941.png', '16227.png', '07957.png', '03097.png', '18689.png', '12785.png', '07015.png', '07080.png', '19432.png', '09163.png', '08769.png', '03349.png', '23651.png', '14109.png', '05916.png', '24057.png', '12622.png', '14069.png', '17623.png', '23567.png', '08988.png', '19997.png', '18468.png', '00846.png', '14062.png', '15622.png', '15780.png', '08151.png', '12716.png', '10191.png', '18351.png', '07003.png', '08515.png', '09215.png', '20263.png', '14244.png', '01148.png', '13321.png', '11509.png', '18175.png', '04066.png', '06303.png', '01898.png', '10083.png', '06831.png', '14027.png', '16334.png', '02048.png', '19923.png', '22894.png', '02569.png', '07023.png', '19553.png', '08142.png', '13368.png', '12958.png', '13447.png', '17798.png', '07199.png', '10606.png', '04448.png', '15075.png', '05321.png', '20624.png', '10949.png', '17184.png', '04970.png', '12558.png', '14483.png', '07875.png', '14608.png', '10077.png', '09563.png', '12850.png', '04588.png', '19939.png', '02484.png', '23322.png', '00222.png', '12682.png', '19229.png', '23454.png', '11399.png', '21708.png', '16586.png', '06847.png', '23966.png', '09879.png', '15610.png', '01510.png', '18487.png', '01629.png', '03639.png', '09714.png', '17442.png', '22406.png', '24687.png', '21548.png', '15222.png', '08709.png', '17328.png', '03380.png', '16415.png', '15777.png', '23296.png', '07682.png', '09672.png', '01050.png', '21685.png', '07328.png', '24771.png', '23748.png', '11318.png', '01649.png', '21261.png', '05495.png', '03212.png', '02982.png', '07241.png', '24217.png', '22732.png', '24273.png', '15762.png', '24537.png', '05059.png', '00892.png', '01271.png', '09325.png', '04385.png', '14727.png', '01676.png', '10180.png', '13335.png', '21560.png', '09896.png', '06472.png', '11040.png', '01157.png', '23969.png', '14319.png', '14476.png', '14632.png', '13529.png', '10204.png', '01098.png', '03776.png', '21775.png', '09161.png', '18446.png', '21786.png', '17654.png', '11352.png', '03607.png', '11655.png', '11799.png', '06063.png', '09913.png', '05649.png', '01358.png', '21135.png', '01439.png', '12349.png', '24401.png', '19319.png', '01636.png', '00546.png', '23194.png', '05974.png', '23282.png', '15887.png', '24314.png', '23471.png', '03847.png', '18397.png', '00107.png', '13781.png', '08475.png', '12394.png', '19909.png', '03621.png', '04353.png', '10325.png', '09202.png', '21274.png', '20828.png', '03414.png', '23043.png', '13200.png', '17133.png', '09120.png', '18944.png', '20312.png', '24047.png', '04770.png', '13287.png', '18046.png', '02377.png', '14239.png', '02354.png', '21583.png', '11417.png', '23189.png', '20806.png', '07528.png', '23023.png', '03450.png', '23156.png', '08250.png', '16822.png', '05588.png', '07198.png', '11361.png', '19277.png', '20654.png', '06924.png', '02385.png', '02254.png', '08284.png', '19210.png', '13198.png', '14530.png', '07384.png', '09927.png', '04008.png', '12629.png', '09206.png', '01035.png', '10485.png', '24197.png', '07467.png', '07743.png', '13483.png', '01437.png', '00733.png', '05601.png', '06339.png', '02942.png', '21901.png', '19531.png', '17455.png', '07847.png', '21041.png', '03413.png', '07030.png', '24651.png', '20535.png', '13061.png', '09200.png', '15972.png', '17994.png', '03729.png', '23892.png', '18681.png', '02444.png', '15563.png', '23511.png', '19387.png', '11769.png', '23284.png', '12540.png', '06608.png', '17410.png', '13585.png', '05435.png', '22476.png', '11569.png', '13077.png', '04666.png', '19318.png', '03313.png', '03836.png', '06388.png', '22348.png', '12208.png', '09473.png', '04544.png', '20338.png', '01280.png', '08221.png', '12151.png', '01879.png', '16715.png', '11473.png', '04979.png', '09366.png', '02964.png', '05002.png', '07537.png', '21407.png', '23738.png', '06826.png', '20171.png', '20173.png', '21878.png', '07746.png', '08992.png', '08488.png', '22585.png', '10623.png', '08448.png', '15533.png', '20130.png', '06014.png', '24133.png', '06910.png', '00849.png', '01277.png', '03245.png', '17582.png', '16964.png', '19239.png', '16912.png', '11017.png', '05593.png', '18263.png', '18842.png', '08122.png', '04456.png', '14194.png', '06309.png', '06318.png', '18014.png', '14112.png', '05168.png', '07727.png', '04984.png', '20881.png', '04950.png', '06168.png', '21699.png', '09643.png', '04340.png', '00805.png', '10528.png', '19213.png', '06595.png', '03855.png', '13352.png', '15356.png', '17763.png', '07204.png', '15173.png', '08882.png', '03699.png', '17890.png', '21741.png', '02026.png', '06806.png', '04634.png', '20830.png', '24188.png', '02211.png', '15833.png', '05665.png', '10925.png', '02644.png', '05297.png', '07381.png', '08730.png', '04186.png', '02025.png', '22300.png', '18192.png', '00806.png', '08342.png', '02231.png', '04974.png', '14723.png', '02463.png', '19338.png', '00045.png', '02353.png', '14852.png', '18523.png', '05620.png', '16515.png', '14828.png', '00429.png', '15644.png', '11663.png', '24079.png', '24104.png', '05472.png', '21899.png', '15883.png', '02778.png', '01969.png', '08426.png', '15191.png', '14438.png', '01746.png', '19854.png', '20435.png', '24539.png', '22077.png', '08136.png', '08568.png', '09937.png', '08093.png', '11823.png', '22502.png', '03566.png', '12801.png', '05304.png', '08952.png', '17990.png', '22836.png', '05008.png', '20054.png', '13871.png', '13083.png', '02720.png', '20044.png', '19872.png', '12587.png', '13748.png', '23329.png', '18259.png', '00145.png', '07009.png', '03273.png', '07687.png', '04805.png', '06354.png', '17026.png', '16700.png', '13079.png', '23096.png', '00040.png', '02259.png', '02691.png', '15616.png', '00274.png', '15475.png', '21254.png', '18688.png', '08505.png', '10390.png', '06695.png', '19340.png', '23734.png', '15879.png', '05412.png', '07815.png', '21821.png', '14857.png', '18659.png', '24361.png', '10477.png', '23246.png', '08070.png', '08887.png', '02996.png', '20966.png', '00002.png', '05335.png', '00704.png', '03572.png', '01970.png', '07919.png', '05205.png', '13783.png', '19601.png', '01024.png', '16068.png', '04343.png', '01446.png', '04476.png', '04794.png', '16401.png', '06834.png', '17657.png', '06441.png', '13372.png', '12765.png', '07939.png', '04250.png', '18805.png', '20642.png', '12080.png', '06825.png', '16434.png', '04962.png', '11358.png', '08143.png', '13332.png', '18466.png', '05140.png', '09192.png', '21447.png', '20974.png', '12909.png', '01876.png', '04985.png', '16671.png', '07363.png', '08170.png', '19293.png', '17905.png', '07226.png', '19085.png', '20603.png', '07286.png', '24006.png', '14242.png', '11694.png', '07677.png', '10976.png', '15476.png', '07295.png', '15253.png', '15115.png', '04267.png', '19459.png', '08575.png', '16403.png', '05711.png', '09316.png', '12798.png', '17965.png', '13063.png', '03294.png', '10192.png', '18646.png', '17744.png', '08352.png', '14198.png', '23446.png', '02868.png', '06194.png', '12556.png', '17527.png', '22404.png', '03043.png', '24215.png', '19735.png', '06169.png', '14662.png', '03046.png', '22005.png', '07672.png', '17997.png', '11930.png', '20896.png', '19537.png', '22617.png', '17204.png', '22363.png', '02045.png', '14849.png', '08367.png', '14234.png', '19411.png', '18647.png', '00174.png', '20614.png', '14830.png', '21175.png', '23779.png', '18419.png', '01342.png', '22384.png', '18067.png', '15770.png', '05044.png', '11093.png', '04627.png', '10126.png', '16992.png', '05658.png', '21942.png', '22591.png', '10140.png', '01931.png', '01455.png', '03312.png', '12904.png', '05259.png', '24496.png', '12340.png', '13251.png', '24693.png', '11111.png', '01572.png', '03735.png', '24015.png', '05015.png', '13280.png', '08743.png', '14736.png', '04147.png', '24613.png', '04730.png', '15941.png', '05615.png', '16410.png', '24532.png', '14083.png', '11531.png', '08824.png', '19335.png', '08816.png', '04941.png', '24311.png', '22301.png', '06935.png', '09485.png', '17214.png', '09129.png', '17154.png', '23648.png', '20633.png', '18032.png', '13686.png', '01832.png', '14594.png', '09476.png', '04908.png', '09319.png', '00848.png', '17879.png', '08038.png', '01807.png', '09767.png', '13831.png', '02283.png', '09741.png', '13174.png', '03348.png', '02519.png', '04816.png', '15895.png', '05723.png', '19593.png', '04133.png', '01659.png', '02419.png', '17608.png', '24007.png', '10625.png', '10273.png', '00541.png', '14190.png', '04896.png', '19841.png', '01378.png', '14678.png', '13101.png', '07588.png', '08042.png', '01893.png', '23254.png', '09220.png', '15431.png', '06207.png', '14654.png', '22399.png', '22223.png', '09941.png', '03410.png', '19355.png', '04411.png', '05614.png', '05220.png', '09064.png', '01858.png', '12640.png', '02747.png', '18635.png', '06325.png', '21287.png', '11317.png', '22589.png', '20399.png', '03394.png', '07645.png', '15387.png', '02078.png', '22251.png', '24593.png', '06726.png', '09181.png', '05770.png', '08443.png', '14748.png', '20818.png', '10072.png', '12114.png', '00468.png', '04256.png', '17618.png', '14161.png', '08312.png', '01875.png', '03140.png', '08703.png', '15355.png', '04484.png', '11709.png', '04656.png', '08671.png', '13611.png', '21109.png', '19017.png', '17385.png', '17522.png', '24070.png', '23650.png', '12383.png', '12201.png', '02355.png', '05744.png', '02474.png', '06681.png', '05694.png', '14789.png', '11312.png', '02022.png', '16275.png', '02756.png', '08993.png', '24263.png', '24568.png', '14580.png', '09947.png', '06688.png', '06669.png', '09318.png', '23928.png', '14770.png', '24740.png', '04653.png', '08509.png', '14075.png', '15093.png', '02966.png', '06655.png', '02611.png', '01675.png', '21106.png', '13736.png', '17289.png', '09157.png', '22184.png', '19109.png', '03584.png', '15797.png', '00358.png', '12369.png', '11067.png', '06543.png', '15948.png', '15192.png', '02607.png', '13286.png', '14238.png', '15107.png', '07679.png', '24857.png', '24848.png', '07516.png', '08653.png', '11851.png', '07202.png', '20148.png', '03887.png', '10004.png', '05650.png', '16077.png', '24430.png', '23093.png', '17398.png', '19100.png', '16058.png', '14792.png', '06931.png', '14420.png', '20250.png', '05838.png', '20619.png', '02495.png', '17953.png', '24535.png', '00609.png', '08186.png', '10573.png', '02248.png', '23297.png', '18701.png', '06030.png', '09438.png', '08160.png', '08855.png', '08827.png', '24869.png', '11005.png', '10196.png', '00528.png', '23143.png', '11625.png', '10159.png', '09702.png', '10398.png', '21550.png', '21216.png', '14767.png', '15452.png', '01246.png', '03770.png', '06352.png', '15303.png', '18365.png', '02835.png', '19679.png', '21324.png', '03636.png', '24181.png', '09141.png', '01100.png', '02391.png', '09441.png', '11803.png', '22053.png', '21001.png', '20994.png', '22914.png', '20508.png', '06586.png', '11228.png', '19715.png', '01963.png', '16331.png', '04909.png', '21849.png', '06941.png', '12527.png', '14560.png', '20760.png', '20320.png', '21980.png', '06547.png', '13128.png', '09229.png', '00558.png', '01321.png', '18896.png', '04833.png', '09511.png', '11492.png', '01767.png', '04918.png', '20464.png', '06715.png', '09637.png', '05830.png', '15503.png', '06957.png', '22778.png', '17764.png', '19597.png', '22388.png', '02998.png', '05440.png', '12770.png', '11645.png', '10522.png', '21435.png', '18445.png', '12749.png', '14197.png', '21604.png', '17878.png', '16095.png', '12237.png', '20931.png', '15209.png', '03888.png', '05835.png', '07715.png', '05006.png', '06435.png', '04523.png', '03157.png', '06216.png', '19199.png', '09959.png', '19224.png', '23728.png', '07514.png', '16690.png', '18903.png', '06602.png', '00680.png', '17853.png', '11602.png', '18064.png', '15775.png', '24880.png', '07899.png', '01518.png', '05118.png', '04831.png', '19190.png', '24933.png', '03256.png', '02965.png', '10007.png', '00457.png', '06734.png', '06323.png', '21806.png', '16453.png', '06123.png', '15338.png', '04757.png', '23561.png', '14504.png', '09051.png', '19933.png', '07711.png', '14171.png', '23483.png', '23201.png', '03162.png', '09742.png', '19730.png', '09053.png', '03876.png', '09226.png', '12003.png', '08087.png', '12649.png', '09117.png', '00777.png', '06346.png', '23037.png', '06202.png', '10396.png', '01735.png', '08163.png', '22148.png', '18783.png', '07660.png', '06962.png', '13737.png', '23874.png', '08695.png', '09805.png', '00458.png', '08498.png', '19698.png', '02791.png', '03978.png', '11822.png', '08218.png', '02893.png', '16426.png', '03547.png', '23123.png', '21446.png', '02157.png', '12495.png', '09155.png', '11624.png', '23280.png', '15162.png', '16226.png', '16307.png', '13502.png', '03121.png', '19711.png', '00989.png', '21105.png', '19222.png', '22000.png', '11731.png', '17765.png', '23279.png', '18265.png', '03523.png', '00944.png', '12989.png', '19037.png', '10266.png', '16369.png', '15928.png', '24471.png', '14763.png', '17038.png', '13683.png', '20322.png', '09624.png', '14374.png', '14125.png', '05994.png', '21783.png', '20868.png', '13804.png', '18795.png', '17340.png', '10929.png', '18392.png', '14322.png', '20021.png', '03672.png', '03848.png', '17362.png', '17792.png', '21571.png', '20394.png', '01643.png', '22079.png', '16428.png', '00605.png', '19648.png', '18749.png', '12376.png', '12910.png', '21703.png', '12637.png', '24507.png', '16361.png', '01359.png', '07529.png', '01564.png', '21217.png', '14943.png', '04278.png', '17074.png', '13825.png', '15317.png', '04864.png', '07252.png', '07713.png', '10123.png', '16634.png', '14066.png', '22643.png', '11237.png', '13844.png', '21705.png', '16669.png', '20191.png', '07532.png', '15120.png', '22710.png', '20580.png', '21661.png', '11735.png']
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

        if i_iter == 254276:
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
