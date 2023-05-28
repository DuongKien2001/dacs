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
    a = ['10975.png', '22699.png', '18009.png', '21976.png', '16123.png', '21388.png', '06048.png', '14661.png', '12683.png', '08394.png', '19631.png', '12518.png', '13740.png', '20700.png', '07441.png', '16603.png', '08398.png', '06789.png', '08879.png', '11566.png', '15914.png', '12258.png', '04452.png', '14473.png', '16442.png', '13762.png', '23789.png', '17844.png', '10166.png', '11543.png', '02321.png', '13582.png', '23246.png', '20405.png', '08589.png', '07461.png', '16896.png', '24786.png', '23454.png', '23873.png', '24177.png', '23818.png', '13196.png', '07623.png', '15979.png', '11742.png', '01611.png', '00865.png', '10845.png', '01724.png', '22454.png', '12657.png', '10967.png', '02840.png', '24934.png', '19678.png', '07449.png', '17628.png', '15012.png', '23819.png', '17925.png', '08663.png', '23750.png', '12514.png', '22763.png', '16985.png', '16144.png', '17845.png', '02879.png', '03370.png', '17972.png', '22460.png', '20520.png', '18215.png', '18196.png', '22216.png', '17086.png', '04110.png', '24154.png', '21699.png', '13070.png', '18094.png', '17288.png', '13846.png', '04053.png', '07028.png', '10783.png', '20502.png', '12061.png', '05332.png', '07223.png', '01546.png', '16923.png', '06597.png', '23003.png', '03443.png', '23765.png', '23465.png', '17955.png', '17203.png', '04719.png', '13219.png', '12021.png', '07227.png', '00959.png', '18512.png', '02380.png', '17065.png', '02374.png', '12151.png', '03151.png', '16895.png', '22309.png', '20532.png', '03119.png', '18069.png', '15191.png', '04113.png', '11022.png', '19564.png', '18540.png', '14596.png', '09981.png', '23766.png', '00329.png', '04982.png', '19885.png', '21545.png', '08361.png', '16953.png', '18064.png', '15978.png', '11365.png', '22339.png', '06167.png', '20774.png', '23616.png', '03232.png', '04508.png', '23226.png', '21345.png', '11601.png', '20521.png', '11127.png', '22211.png', '19126.png', '09464.png', '02296.png', '24868.png', '13085.png', '05236.png', '22533.png', '24417.png', '14237.png', '22262.png', '22233.png', '21907.png', '01247.png', '17539.png', '18167.png', '24609.png', '23612.png', '23384.png', '15963.png', '05438.png', '15661.png', '23979.png', '03019.png', '06021.png', '02794.png', '08633.png', '01341.png', '11618.png', '10906.png', '06996.png', '10501.png', '21531.png', '13925.png', '21663.png', '20886.png', '03621.png', '19519.png', '13281.png', '23301.png', '08592.png', '23534.png', '10788.png', '06219.png', '09048.png', '01018.png', '18072.png', '22425.png', '10688.png', '12993.png', '19915.png', '04964.png', '22078.png', '19604.png', '17866.png', '23861.png', '01012.png', '23804.png', '11105.png', '24818.png', '12055.png', '10400.png', '16865.png', '23925.png', '07405.png', '06391.png', '12426.png', '24300.png', '08529.png', '24188.png', '08363.png', '18417.png', '07865.png', '15080.png', '08709.png', '14389.png', '02150.png', '13526.png', '14167.png', '22284.png', '02859.png', '13408.png', '18711.png', '11189.png', '01702.png', '13666.png', '11877.png', '20874.png', '24815.png', '02016.png', '06392.png', '02125.png', '15915.png', '16343.png', '09614.png', '10070.png', '19904.png', '09993.png', '03596.png', '10041.png', '01646.png', '09386.png', '23658.png', '11761.png', '00542.png', '17915.png', '11195.png', '20633.png', '21337.png', '22618.png', '18523.png', '06294.png', '21634.png', '02513.png', '20695.png', '08638.png', '12820.png', '15859.png', '17139.png', '09624.png', '21210.png', '09374.png', '13889.png', '20061.png', '09455.png', '01042.png', '13730.png', '21223.png', '07634.png', '02523.png', '15560.png', '18858.png', '13221.png', '20680.png', '09504.png', '18118.png', '02105.png', '11345.png', '08880.png', '00939.png', '13991.png', '07944.png', '00591.png', '05425.png', '16409.png', '24349.png', '10916.png', '00176.png', '11268.png', '02704.png', '00289.png', '00016.png', '11753.png', '05562.png', '21566.png', '11905.png', '04822.png', '08753.png', '11645.png', '05747.png', '20511.png', '03072.png', '17272.png', '12801.png', '20329.png', '05282.png', '14436.png', '22901.png', '08893.png', '06212.png', '11300.png', '23893.png', '04567.png', '23837.png', '22548.png', '10129.png', '04093.png', '07024.png', '15154.png', '11880.png', '09151.png', '17244.png', '24445.png', '14172.png', '14003.png', '10097.png', '00456.png', '04684.png', '19132.png', '06208.png', '06926.png', '22250.png', '09069.png', '00589.png', '09736.png', '06414.png', '20019.png', '01802.png', '08855.png', '08586.png', '00973.png', '06948.png', '22843.png', '01249.png', '15741.png', '19458.png', '24365.png', '21680.png', '03585.png', '14716.png', '02024.png', '12126.png', '01210.png', '18537.png', '01130.png', '21434.png', '00932.png', '01756.png', '22928.png', '18853.png', '19786.png', '09028.png', '24937.png', '02886.png', '21482.png', '15376.png', '19758.png', '11246.png', '19796.png', '21711.png', '00647.png', '17576.png', '06939.png', '20226.png', '09102.png', '23427.png', '04633.png', '08502.png', '20404.png', '23459.png', '23030.png', '00706.png', '02165.png', '21443.png', '15846.png', '09922.png', '23477.png', '22469.png', '02851.png', '15909.png', '20115.png', '07429.png', '20003.png', '18271.png', '20408.png', '09590.png', '23142.png', '13605.png', '22557.png', '04166.png', '08457.png', '11128.png', '16224.png', '19408.png', '19747.png', '07443.png', '05536.png', '12152.png', '12227.png', '19616.png', '05751.png', '06799.png', '23446.png', '18615.png', '13869.png', '15922.png', '05065.png', '12818.png', '12741.png', '18748.png', '05260.png', '21974.png', '21713.png', '07552.png', '08525.png', '24853.png', '20634.png', '12736.png', '20493.png', '06013.png', '07265.png', '08270.png', '04164.png', '14448.png', '19308.png', '15149.png', '16473.png', '16051.png', '22821.png', '10868.png', '22083.png', '05889.png', '03106.png', '04774.png', '07167.png', '23666.png', '07117.png', '01234.png', '08017.png', '16325.png', '08937.png', '12833.png', '09296.png', '13390.png', '13124.png', '11234.png', '15237.png', '23692.png', '20779.png', '24953.png', '22812.png', '00575.png', '23339.png', '18109.png', '01577.png', '11844.png', '17880.png', '23128.png', '23740.png', '22024.png', '21416.png', '15048.png', '13174.png', '00806.png', '12609.png', '12624.png', '16321.png', '09394.png', '24204.png', '10612.png', '12863.png', '07311.png', '22371.png', '11994.png', '04925.png', '20498.png', '07896.png', '22769.png', '20994.png', '05373.png', '18017.png', '16802.png', '16949.png', '02948.png', '16600.png', '02514.png', '11534.png', '19204.png', '01225.png', '12333.png', '24713.png', '14339.png', '14720.png', '02365.png', '11629.png', '20575.png', '11779.png', '19903.png', '07326.png', '09811.png', '14064.png', '21245.png', '07674.png', '01223.png', '06639.png', '00046.png', '16791.png', '19974.png', '17564.png', '14856.png', '15269.png', '15612.png', '08351.png', '08373.png', '17402.png', '21930.png', '02691.png', '22103.png', '17344.png', '01627.png', '15291.png', '06819.png', '15030.png', '24084.png', '10624.png', '11057.png', '18779.png', '18534.png', '23314.png', '22462.png', '05869.png', '01718.png', '12823.png', '04018.png', '00720.png', '13797.png', '10699.png', '24736.png', '23190.png', '06015.png', '03118.png', '19779.png', '02562.png', '04692.png', '02890.png', '02923.png', '24883.png', '01929.png', '17114.png', '07459.png', '15779.png', '02707.png', '20077.png', '16509.png', '13035.png', '00814.png', '10789.png', '22895.png', '17850.png', '08395.png', '13008.png', '08304.png', '07255.png', '21273.png', '10953.png', '11438.png', '05791.png', '23881.png', '15576.png', '03066.png', '14678.png', '06836.png', '15333.png', '00217.png', '09287.png', '01453.png', '10530.png', '08722.png', '03373.png', '15582.png', '01730.png', '00966.png', '05302.png', '03206.png', '14894.png', '00093.png', '22170.png', '19793.png', '15754.png', '13713.png', '19756.png', '00628.png', '16511.png', '05013.png', '07551.png', '13161.png', '11513.png', '22790.png', '10933.png', '12037.png', '05544.png', '04327.png', '05783.png', '00151.png', '09258.png', '19047.png', '21022.png', '19681.png', '21787.png', '00075.png', '03461.png', '03811.png', '11869.png', '22690.png', '07433.png', '08916.png', '06709.png', '23540.png', '13788.png', '20363.png', '13484.png', '14555.png', '21990.png', '12805.png', '15926.png', '04354.png', '21335.png', '09702.png', '03162.png', '24724.png', '04482.png', '06354.png', '01763.png', '07931.png', '15498.png', '13617.png', '23407.png', '00291.png', '17163.png', '09569.png', '01760.png', '13228.png', '21561.png', '08161.png', '11262.png', '09478.png', '18964.png', '19288.png', '00559.png', '08453.png', '09845.png', '16624.png', '09657.png', '13657.png', '15476.png', '11167.png', '05162.png', '18938.png', '00228.png', '23694.png', '10737.png', '03324.png', '02329.png', '09378.png', '06854.png', '15342.png', '01629.png', '20422.png', '07394.png', '06613.png', '05244.png', '23527.png', '04845.png', '06037.png', '01822.png', '12575.png', '00927.png', '05654.png', '11010.png', '02864.png', '15614.png', '19523.png', '24196.png', '09938.png', '13300.png', '10671.png', '22883.png', '17957.png', '00092.png', '11098.png', '20800.png', '13556.png', '15306.png', '19536.png', '04209.png', '23650.png', '05852.png', '11368.png', '12097.png', '11248.png', '17355.png', '24697.png', '16156.png', '10356.png', '09681.png', '23085.png', '18880.png', '04856.png', '08810.png', '07043.png', '18024.png', '24315.png', '23576.png', '24538.png', '07454.png', '21951.png', '20791.png', '02498.png', '04871.png', '00210.png', '02541.png', '08061.png', '20033.png', '20627.png', '20064.png', '18815.png', '24861.png', '18360.png', '24107.png', '19286.png', '17578.png', '17658.png', '09467.png', '09862.png', '19150.png', '17285.png', '12260.png', '03441.png', '21657.png', '07628.png', '18642.png', '09126.png', '15919.png', '05312.png', '17971.png', '23198.png', '00014.png', '21251.png', '11712.png', '02230.png', '18592.png', '07400.png', '22916.png', '16205.png', '10745.png', '19783.png', '21927.png', '18043.png', '21863.png', '02073.png', '01107.png', '02448.png', '06828.png', '01785.png', '11893.png', '14805.png', '10763.png', '13813.png', '04804.png', '03694.png', '19234.png', '04350.png', '06961.png', '15298.png', '21510.png', '24286.png', '04008.png', '16610.png', '21272.png', '04050.png', '12313.png', '22965.png', '01299.png', '09255.png', '07586.png', '01499.png', '19201.png', '08156.png', '07468.png', '16198.png', '19453.png', '03255.png', '05598.png', '22209.png', '03318.png', '14068.png', '07630.png', '10043.png', '19076.png', '11184.png', '24711.png', '07560.png', '04968.png', '02624.png', '17714.png', '14413.png', '21192.png', '21900.png', '15177.png', '18323.png', '13787.png', '04685.png', '02017.png', '17269.png', '00482.png', '10205.png', '21195.png', '09502.png', '15482.png', '04728.png', '05673.png', '22274.png', '22356.png', '05694.png', '24203.png', '15472.png', '15140.png', '16998.png', '09067.png', '05492.png', '21143.png', '08291.png', '13519.png', '14789.png', '07022.png', '16529.png', '24220.png', '01995.png', '20477.png', '06889.png', '18477.png', '04005.png', '10369.png', '11621.png', '01693.png', '20187.png', '19789.png', '05230.png', '22987.png', '01686.png', '19657.png', '01712.png', '06879.png', '24819.png', '04909.png', '22896.png', '07643.png', '11104.png', '21786.png', '14631.png', '12387.png', '03390.png', '09220.png', '02004.png', '18392.png', '22590.png', '09921.png', '21581.png', '05114.png', '04590.png', '04312.png', '12777.png', '19635.png', '23715.png', '16331.png', '15104.png', '22421.png', '16300.png', '04181.png', '09962.png', '07912.png', '02913.png', '10355.png', '14733.png', '14411.png', '06348.png', '06271.png', '15210.png', '19175.png', '21840.png', '12720.png', '09314.png', '00204.png', '22281.png', '21811.png', '23558.png', '11882.png', '14143.png', '06977.png', '15494.png', '05055.png', '09180.png', '24516.png', '20891.png', '02912.png', '20270.png', '15771.png', '01518.png', '23632.png', '07281.png', '22067.png', '21295.png', '07995.png', '18631.png', '21396.png', '00817.png', '08296.png', '05721.png', '17206.png', '07840.png', '18138.png', '02776.png', '17827.png', '05882.png', '09561.png', '20091.png', '21688.png', '01886.png', '15081.png', '13028.png', '23512.png', '19197.png', '12999.png', '06273.png', '16395.png', '17802.png', '02856.png', '14963.png', '24945.png', '23251.png', '06678.png', '00863.png', '22767.png', '08366.png', '07662.png', '23680.png', '21571.png', '09496.png', '23891.png', '15313.png', '00666.png', '05426.png', '19495.png', '08820.png', '10025.png', '00898.png', '12578.png', '14122.png', '04658.png', '19687.png', '18700.png', '22232.png', '14188.png', '05025.png', '20927.png', '00740.png', '15824.png', '17873.png', '08742.png', '17871.png', '19439.png', '16595.png', '14019.png', '03412.png', '16580.png', '06455.png', '19574.png', '01406.png', '09824.png', '22701.png', '01250.png', '24351.png', '06932.png', '00540.png', '02019.png', '06973.png', '06238.png', '19502.png', '14606.png', '01028.png', '19754.png', '10784.png', '02627.png', '13200.png', '15770.png', '05265.png', '24117.png', '05973.png', '01268.png', '17406.png', '10905.png', '07933.png', '03407.png', '06247.png', '24249.png', '23476.png', '04678.png', '17189.png', '23652.png', '14236.png', '22958.png', '05664.png', '20816.png', '02635.png', '04046.png', '15225.png', '00301.png', '08983.png', '01542.png', '08979.png', '02118.png', '02345.png', '19048.png', '00410.png', '15804.png', '04704.png', '13278.png', '14367.png', '19478.png', '10418.png', '06157.png', '10479.png', '10706.png', '04030.png', '20622.png', '09974.png', '01595.png', '01713.png', '06496.png', '11498.png', '13031.png', '05214.png', '05762.png', '23388.png', '21227.png', '13352.png', '13406.png', '20319.png', '22005.png', '10637.png', '07072.png', '07993.png', '06901.png', '15874.png', '06718.png', '15122.png', '11117.png', '19017.png', '22666.png', '00810.png', '23846.png', '20068.png', '02048.png', '11609.png', '09451.png', '04191.png', '23670.png', '21231.png', '13614.png', '13744.png', '10382.png', '02995.png', '03236.png', '07838.png', '16536.png', '16712.png', '19923.png', '08060.png', '14739.png', '03250.png', '16237.png', '05503.png', '05542.png', '08149.png', '06519.png', '24822.png', '23373.png', '22818.png', '24773.png', '08508.png', '04167.png', '06188.png', '02918.png', '16232.png', '23551.png', '13557.png', '14911.png', '05940.png', '01414.png', '06606.png', '04320.png', '18787.png', '18099.png', '23391.png', '00711.png', '20258.png', '13833.png', '08634.png', '05493.png', '18497.png', '24803.png', '16212.png', '03272.png', '21888.png', '02649.png', '00630.png', '23071.png', '20058.png', '19389.png', '04017.png', '05486.png', '02203.png', '08941.png', '13057.png', '11359.png', '16809.png', '18370.png', '01370.png', '01666.png', '14412.png', '18544.png', '14569.png', '08412.png', '19473.png', '14835.png', '01441.png', '02367.png', '16171.png', '00568.png', '11016.png', '10244.png', '14381.png', '12351.png', '08630.png', '21619.png', '08084.png', '02477.png', '15752.png', '07539.png', '16175.png', '18622.png', '11961.png', '21835.png', '04198.png', '19359.png', '02616.png', '03295.png', '13786.png', '08695.png', '01750.png', '24371.png', '05674.png', '20013.png', '19069.png', '07067.png', '20503.png', '19101.png', '04072.png', '22932.png', '21159.png', '18329.png', '10392.png', '16022.png', '23468.png', '22498.png', '10525.png', '21558.png', '03363.png', '12303.png', '07909.png', '03201.png', '19225.png', '16091.png', '13726.png', '04232.png', '08164.png', '09853.png', '01786.png', '15193.png', '22785.png', '09526.png', '05861.png', '03251.png', '08857.png', '16143.png', '18335.png', '20204.png', '19969.png', '16987.png', '21839.png', '00439.png', '05221.png', '11304.png', '11425.png', '08664.png', '12094.png', '24833.png', '06111.png', '00050.png', '24686.png', '24526.png', '15389.png', '18348.png', '22199.png', '07140.png', '02531.png', '15984.png', '12659.png', '20088.png', '00084.png', '18992.png', '04527.png', '00860.png', '06024.png', '21890.png', '04604.png', '00908.png', '10063.png', '08579.png', '24273.png', '12987.png', '11770.png', '09700.png', '16046.png', '18363.png', '16849.png', '02035.png', '00997.png', '04866.png', '24517.png', '08961.png', '21794.png', '18295.png', '17863.png', '03910.png', '02947.png', '18582.png', '17700.png', '02202.png', '11512.png', '23093.png', '10232.png', '11820.png', '06311.png', '04605.png', '24502.png', '22706.png', '04792.png', '05530.png', '11205.png', '15452.png', '16786.png', '09058.png', '05615.png', '11561.png', '18282.png', '00581.png', '08737.png', '16316.png', '03416.png', '08677.png', '16658.png', '17524.png', '02669.png', '04493.png', '02344.png', '22003.png', '01564.png', '01956.png', '15274.png', '15765.png', '21511.png', '19140.png', '00574.png', '13815.png', '14647.png', '12052.png', '23557.png', '01348.png', '00033.png', '01056.png', '08040.png', '16248.png', '08966.png', '09990.png', '07533.png', '18141.png', '01219.png', '07204.png', '01458.png', '23148.png', '16989.png', '06965.png', '02222.png', '04942.png', '24523.png', '12503.png', '19054.png', '16472.png', '12265.png', '21490.png', '01080.png', '23345.png', '15524.png', '01573.png', '09844.png', '06482.png', '18319.png', '14084.png', '20296.png', '08846.png', '01659.png', '07036.png', '10644.png', '22607.png', '04014.png', '18545.png', '13988.png', '22077.png', '00653.png', '04980.png', '23442.png', '13863.png', '24038.png', '11705.png', '10500.png', '02739.png', '05983.png', '05045.png', '04997.png', '17443.png', '15940.png', '20350.png', '21887.png', '12713.png', '23505.png', '18535.png', '03466.png', '24630.png', '10668.png', '23377.png', '06582.png', '02901.png', '07949.png', '10544.png', '02985.png', '13292.png', '12935.png', '07287.png', '18244.png', '17173.png', '12023.png', '12585.png', '22739.png', '15246.png', '11090.png', '23537.png', '14423.png', '15835.png', '10510.png', '24589.png', '00974.png', '23263.png', '07001.png', '06520.png', '16191.png', '13768.png', '10384.png', '21202.png', '02070.png', '06621.png', '08116.png', '06779.png', '11061.png', '15430.png', '22133.png', '17543.png', '02841.png', '00499.png', '17751.png', '13611.png', '00534.png', '05154.png', '23860.png', '05651.png', '03907.png', '00319.png', '05142.png', '15253.png', '13052.png', '09141.png', '01192.png', '13056.png', '21763.png', '07469.png', '23509.png', '17437.png', '08422.png', '00086.png', '20093.png', '02573.png', '07692.png', '19066.png', '20462.png', '18388.png', '00240.png', '02695.png', '00822.png', '17201.png', '05278.png', '13777.png', '21168.png', '18864.png', '10665.png', '13884.png', '10931.png', '19117.png', '13108.png', '05092.png', '22161.png', '04600.png', '20848.png', '04012.png', '21503.png', '13243.png', '15501.png', '05309.png', '06110.png', '06886.png', '03223.png', '15676.png', '21362.png', '09646.png', '00831.png', '16253.png', '18557.png', '22998.png', '23577.png', '24635.png', '24845.png', '07753.png', '18090.png', '16794.png', '18602.png', '07363.png', '20660.png', '10248.png', '20480.png', '01424.png', '23395.png', '12584.png', '21813.png', '09756.png', '15241.png', '23843.png', '12211.png', '03802.png', '16079.png', '16163.png', '12995.png', '23787.png', '10256.png', '11306.png', '12446.png', '11691.png', '10177.png', '21142.png', '15942.png', '21917.png', '06992.png', '15308.png', '15230.png', '24455.png', '11687.png', '17068.png', '03309.png', '14867.png', '18885.png', '13533.png', '06990.png', '20275.png', '15022.png', '15063.png', '23585.png', '11838.png', '02961.png', '17315.png', '07342.png', '24122.png', '18406.png', '22676.png', '13707.png', '03922.png', '10633.png', '19720.png', '05163.png', '15039.png', '04474.png', '17174.png', '10085.png', '15711.png', '10847.png', '20260.png', '04897.png', '17654.png', '10127.png', '21201.png', '20747.png', '08658.png', '05149.png', '21259.png', '10296.png', '17202.png', '16577.png', '06145.png', '21748.png', '08731.png', '10112.png', '03693.png', '01576.png', '10278.png', '17696.png', '21782.png', '08714.png', '20286.png', '21014.png', '11350.png', '07953.png', '24779.png', '00497.png', '17616.png', '18154.png', '24743.png', '16684.png', '23871.png', '12204.png', '15917.png', '02984.png', '06839.png', '12547.png', '14263.png', '16235.png', '07492.png', '15495.png', '06747.png', '19778.png', '15051.png', '05097.png', '18546.png', '13442.png', '10153.png', '04801.png', '23309.png', '15552.png', '11854.png', '16961.png', '20163.png', '19106.png', '01222.png', '14057.png', '02177.png', '06408.png', '01594.png', '22887.png', '16626.png', '10106.png', '12084.png', '01448.png', '10405.png', '16993.png', '13752.png', '24181.png', '07251.png', '19559.png', '21831.png', '01230.png', '08004.png', '14288.png', '10133.png', '17982.png', '10529.png', '11738.png', '21418.png', '21737.png', '00547.png', '24840.png', '22304.png', '21683.png', '12237.png', '24763.png', '19822.png', '23634.png', '20755.png', '11418.png', '18062.png', '22797.png', '21426.png', '19280.png', '20670.png', '08010.png', '03384.png', '05399.png', '09447.png', '05565.png', '08261.png', '00603.png', '02974.png', '23782.png', '03216.png', '02690.png', '05690.png', '07695.png', '01331.png', '16180.png', '23600.png', '18703.png', '14642.png', '14626.png', '23403.png', '01098.png', '12699.png', '03697.png', '22249.png', '19378.png', '11377.png', '08143.png', '14097.png', '19528.png', '20793.png', '17677.png', '11254.png', '12441.png', '23303.png', '19205.png', '00998.png', '12241.png', '02395.png', '24352.png', '06393.png', '07996.png', '04178.png', '12378.png', '02998.png', '14415.png', '20220.png', '17101.png', '22485.png', '10128.png', '04259.png', '07360.png', '23299.png', '05351.png', '20972.png', '09076.png', '15687.png', '24946.png', '03959.png', '03892.png', '16543.png', '07047.png', '16150.png', '19123.png', '07408.png', '02195.png', '05014.png', '17752.png', '20912.png', '11410.png', '06233.png', '22879.png', '24433.png', '06340.png', '23074.png', '01544.png', '04300.png', '11899.png', '08809.png', '02792.png', '20850.png', '12032.png', '07515.png', '01440.png', '00091.png', '16597.png', '22288.png', '08933.png', '05566.png', '09542.png', '12024.png', '02325.png', '07396.png', '03787.png', '00147.png', '03672.png', '00039.png', '15046.png', '21469.png', '09595.png', '11550.png', '15312.png', '08204.png', '19848.png', '12618.png', '24768.png', '18706.png', '08195.png', '21862.png', '12091.png', '19534.png', '20902.png', '12312.png', '12267.png', '16768.png', '22335.png', '00899.png', '23572.png', '22565.png', '16527.png', '23686.png', '23078.png', '13598.png', '01334.png', '00303.png', '22224.png', '03329.png', '11918.png', '15786.png', '21458.png', '06716.png', '07126.png', '19469.png', '19115.png', '23829.png', '20240.png', '15023.png', '14912.png', '19650.png', '23748.png', '09856.png', '20737.png', '24791.png', '03231.png', '08548.png', '01444.png', '24206.png', '11523.png', '14675.png', '04725.png', '18709.png', '08996.png', '09822.png', '03362.png', '12597.png', '24311.png', '18892.png', '01319.png', '12104.png', '10787.png', '14159.png', '20625.png', '08845.png', '22429.png', '20524.png', '14351.png', '19540.png', '16766.png', '20347.png', '14126.png', '03692.png', '19334.png', '06307.png', '17768.png', '08269.png', '03856.png', '14281.png', '13636.png', '01416.png', '13472.png', '15751.png', '17038.png', '00468.png', '11789.png', '19961.png', '13259.png', '06362.png', '14893.png', '18402.png', '05718.png', '18701.png', '05428.png', '11878.png', '02250.png', '18665.png', '23811.png', '12557.png', '08571.png', '17434.png', '13946.png', '16115.png', '22257.png', '14287.png', '10489.png', '20871.png', '19067.png', '24399.png', '02745.png', '07276.png', '13877.png', '24957.png', '04100.png', '01770.png', '24185.png', '17037.png', '03901.png', '17195.png', '21039.png', '03889.png', '05610.png', '20312.png', '02320.png', '01463.png', '03738.png', '17363.png', '18151.png', '17107.png', '20777.png', '04133.png', '06060.png', '00465.png', '10342.png', '16307.png', '16178.png', '07917.png', '07428.png', '13595.png', '12608.png', '19424.png', '03499.png', '06014.png', '10712.png', '06306.png', '04286.png', '18106.png', '09917.png', '19759.png', '08119.png', '00852.png', '08545.png', '00087.png', '24618.png', '11451.png', '23496.png', '05724.png', '20490.png', '00691.png', '11122.png', '14395.png', '18213.png', '02273.png', '05661.png', '15250.png', '07832.png', '23497.png', '10483.png', '23653.png', '05216.png', '03056.png', '00112.png', '00526.png', '21175.png', '12647.png', '09160.png', '18204.png', '16005.png', '21955.png', '04874.png', '24462.png', '00371.png', '11123.png', '13708.png', '09607.png', '04981.png', '06460.png', '12282.png', '24297.png', '13775.png', '00472.png', '01405.png', '15061.png', '07694.png', '13246.png', '21002.png', '23315.png', '20834.png', '20122.png', '21770.png', '09512.png', '03187.png', '13823.png', '04148.png', '05032.png', '03785.png', '08558.png', '23415.png', '03650.png', '22090.png', '15854.png', '15564.png', '09357.png', '03733.png', '09222.png', '10818.png', '05574.png', '05377.png', '17293.png', '06116.png', '19347.png', '03669.png', '14199.png', '07573.png', '23217.png', '04341.png', '13613.png', '13975.png', '13840.png', '04451.png', '18412.png', '04877.png', '23711.png', '23916.png', '17948.png', '19780.png', '17783.png', '12081.png', '05259.png', '19022.png', '24663.png', '23461.png', '13452.png', '08399.png', '14887.png', '15680.png', '17241.png', '21381.png', '01074.png', '16540.png', '13210.png', '10488.png', '06626.png', '15760.png', '07657.png', '02176.png', '04715.png', '15925.png', '14643.png', '08078.png', '00180.png', '17930.png', '04273.png', '13960.png', '17736.png', '22377.png', '03566.png', '12771.png', '03727.png', '15161.png', '03729.png', '01626.png', '13187.png', '07438.png', '04195.png', '03543.png', '03488.png', '10121.png', '06398.png', '16410.png', '16731.png', '00585.png', '24793.png', '22477.png', '23716.png', '06400.png', '11467.png', '23342.png', '04260.png', '01524.png', '11850.png', '12336.png', '12394.png', '21734.png', '13670.png', '19207.png', '10411.png', '04450.png', '10617.png', '21129.png', '14160.png', '15446.png', '07451.png', '01558.png', '20412.png', '11558.png', '04854.png', '16805.png', '02550.png', '09828.png', '12045.png', '04818.png', '03052.png', '01990.png', '17115.png', '21938.png', '02407.png', '01062.png', '06231.png', '16945.png', '00719.png', '19638.png', '08276.png', '22839.png', '20301.png', '12994.png', '08972.png', '01708.png', '21547.png', '20000.png', '03526.png', '21732.png', '07820.png', '21029.png', '15632.png', '02061.png', '07220.png', '06980.png', '09192.png', '07629.png', '02339.png', '10171.png', '11557.png', '02611.png', '19093.png', '05714.png', '21788.png', '11374.png', '14735.png', '06230.png', '07975.png', '11019.png', '07112.png', '04636.png', '01839.png', '22633.png', '20639.png', '17200.png', '16059.png', '08295.png', '09884.png', '17521.png', '13591.png', '01380.png', '22989.png', '21131.png', '21046.png', '03326.png', '24719.png', '16524.png', '15405.png', '14430.png', '23387.png', '23929.png', '08832.png', '06160.png', '07529.png', '24078.png', '06922.png', '08903.png', '08650.png', '18852.png', '15265.png', '23280.png', '04738.png', '13232.png', '07768.png', '06720.png', '12595.png', '24794.png', '21733.png', '11373.png', '02389.png', '18333.png', '15593.png', '19941.png', '24429.png', '23151.png', '09809.png', '09848.png', '15395.png', '01919.png', '06001.png', '01568.png', '03050.png', '04221.png', '01132.png', '16094.png', '08201.png', '04124.png', '03578.png', '14292.png', '15181.png', '12472.png', '13539.png', '11750.png', '11772.png', '18249.png', '11525.png', '08260.png', '07941.png', '09282.png', '09898.png', '04534.png', '23350.png', '15611.png', '03757.png', '05313.png', '17459.png', '17280.png', '00219.png', '24143.png', '14494.png', '02882.png', '05841.png', '18357.png', '23346.png', '09082.png', '07369.png', '20316.png', '00858.png', '23815.png', '21025.png', '15648.png', '09283.png', '03876.png', '18778.png', '21504.png', '20290.png', '09260.png', '19596.png', '21814.png', '06018.png', '06365.png', '14024.png', '17382.png', '16517.png', '13405.png', '00708.png', '22030.png', '09167.png', '18042.png', '13481.png', '15073.png', '10262.png', '24041.png', '05322.png', '10819.png', '04470.png', '04632.png', '24427.png', '20109.png', '20885.png', '13980.png', '04867.png', '23742.png', '04488.png', '00160.png', '14256.png', '14981.png', '07434.png', '15058.png', '11236.png', '05325.png', '08390.png', '13956.png', '20015.png', '21580.png', '18487.png', '17159.png', '16007.png', '15013.png', '14712.png', '07675.png', '13101.png', '04323.png', '12570.png', '15234.png', '06949.png', '07426.png', '02569.png', '04742.png', '14888.png', '23489.png', '09239.png', '11458.png', '15117.png', '16872.png', '13573.png', '17143.png', '06351.png', '24334.png', '21369.png', '11507.png', '11968.png', '03919.png', '21269.png', '10464.png', '08609.png', '11147.png', '21226.png', '01633.png', '04770.png', '14475.png', '21319.png', '18736.png', '22737.png', '24416.png', '15100.png', '17861.png', '17088.png', '02443.png', '16049.png', '08462.png', '16982.png', '14627.png', '19016.png', '01261.png', '17053.png', '14480.png', '14066.png', '24267.png', '10486.png', '12517.png', '10161.png', '02038.png', '05790.png', '05963.png', '10219.png', '24226.png', '16552.png', '22411.png', '00494.png', '04546.png', '01677.png', '07442.png', '09005.png', '14265.png', '12581.png', '10475.png', '17013.png', '06428.png', '10990.png', '22491.png', '24060.png', '18186.png', '13043.png', '08744.png', '24482.png', '01325.png', '15254.png', '17912.png', '10844.png', '04162.png', '11430.png', '16339.png', '03263.png', '01258.png', '11640.png', '22150.png', '09256.png', '16299.png', '13792.png', '15646.png', '21021.png', '07176.png', '11653.png', '23824.png', '04803.png', '18999.png', '10814.png', '17510.png', '08543.png', '21466.png', '24505.png', '08990.png', '18188.png', '05939.png', '04267.png', '09655.png', '05416.png', '03595.png', '05722.png', '14231.png', '22243.png', '06622.png', '09494.png', '02174.png', '22877.png', '03957.png', '16976.png', '22920.png', '15642.png', '06381.png', '16991.png', '10005.png', '09912.png', '08138.png', '16071.png', '12212.png', '08601.png', '15035.png', '00933.png', '07294.png', '24423.png', '23662.png', '05233.png', '01808.png', '18826.png', '13063.png', '14941.png', '12022.png', '15343.png', '06818.png', '00138.png', '20843.png', '22181.png', '10067.png', '17686.png', '21406.png', '23862.png', '04767.png', '22194.png', '00090.png', '10810.png', '14404.png', '00761.png', '20706.png', '24875.png', '12281.png', '17583.png', '22596.png', '14076.png', '06129.png', '04074.png', '11824.png', '20100.png', '04065.png', '06491.png', '22939.png', '16015.png', '15223.png', '16666.png', '18112.png', '21928.png', '00816.png', '11828.png', '22952.png', '14299.png', '19000.png', '17998.png', '06843.png', '05496.png', '05193.png', '16574.png', '14242.png', '16121.png', '09025.png', '15842.png', '10527.png', '21827.png', '17579.png', '12471.png', '21931.png', '10050.png', '06652.png', '19669.png', '08449.png', '12067.png', '07157.png', '17444.png', '01674.png', '02701.png', '16210.png', '20760.png', '12105.png', '15631.png', '19253.png', '21759.png', '00548.png', '02572.png', '04598.png', '08568.png', '13080.png', '03860.png', '09463.png', '02605.png', '23045.png', '24676.png', '00275.png', '05267.png', '11218.png', '18768.png', '18857.png', '07267.png', '21647.png', '07259.png', '01550.png', '24508.png', '10999.png', '03173.png', '01823.png', '20335.png', '22210.png', '12708.png', '20219.png', '22884.png', '17664.png', '06439.png', '13741.png', '19037.png', '09021.png', '12667.png', '08629.png', '03832.png', '14869.png', '20566.png', '00488.png', '13879.png', '21816.png', '03513.png', '14101.png', '19415.png', '03752.png', '16694.png', '09113.png', '10300.png', '15107.png', '19567.png', '23349.png', '14496.png', '14591.png', '02413.png', '17787.png', '03851.png', '19880.png', '21279.png', '17944.png', '23393.png', '23320.png', '11769.png', '04103.png', '20447.png', '21028.png', '11314.png', '12875.png', '22503.png', '03040.png', '23601.png', '10672.png', '02124.png', '19405.png', '20284.png', '12385.png', '12400.png', '22679.png', '06189.png', '05337.png', '08035.png', '15628.png', '21672.png', '24528.png', '02581.png', '22063.png', '08671.png', '20714.png', '10176.png', '24470.png', '03904.png', '20969.png', '19707.png', '15018.png', '06195.png', '09795.png', '10932.png', '17586.png', '06507.png', '24035.png', '18216.png', '05329.png', '16275.png', '05686.png', '04391.png', '11613.png', '07190.png', '17412.png', '07290.png', '11011.png', '12439.png', '01162.png', '23335.png', '09010.png', '13342.png', '07537.png', '19855.png', '21174.png', '11606.png', '00165.png', '16901.png', '10833.png', '12384.png', '23775.png', '06004.png', '19872.png', '02263.png', '10669.png', '07579.png', '17108.png', '14257.png', '05483.png', '21621.png', '20794.png', '11035.png', '21150.png', '07093.png', '18657.png', '11039.png', '06611.png', '00506.png', '16709.png', '03034.png', '05117.png', '22827.png', '04597.png', '04091.png', '24005.png', '15947.png', '11624.png', '14633.png', '15424.png', '15477.png', '20041.png', '16867.png', '05569.png', '04848.png', '05710.png', '23291.png', '17221.png', '16036.png', '22042.png', '22743.png', '05678.png', '08840.png', '08265.png', '17183.png', '15448.png', '16733.png', '15322.png', '11107.png', '02120.png', '13714.png', '12678.png', '15443.png', '15314.png', '14479.png', '17024.png', '00166.png', '16721.png', '21389.png', '18446.png', '20006.png', '22431.png', '22207.png', '10909.png', '22911.png', '12080.png', '07101.png', '23997.png', '15449.png', '05134.png', '24639.png', '05359.png', '16282.png', '22968.png', '20470.png', '20931.png', '13306.png', '24173.png', '10381.png', '06044.png', '12364.png', '03906.png', '06213.png', '03659.png', '11910.png', '09609.png', '17638.png', '11997.png', '03918.png', '13354.png', '14996.png', '16508.png', '15783.png', '20761.png', '18386.png', '18682.png', '16742.png', '14333.png', '07722.png', '01478.png', '06451.png', '23328.png', '20708.png', '15404.png', '19208.png', '18670.png', '08355.png', '23569.png', '20663.png', '13431.png', '07160.png', '13958.png', '23690.png', '14586.png', '14897.png', '09216.png', '22278.png', '02251.png', '17415.png', '10243.png', '10941.png', '14226.png', '17263.png', '10732.png', '03935.png', '03619.png', '03993.png', '02373.png', '02031.png', '19261.png', '19229.png', '00380.png', '05560.png', '22547.png', '05761.png', '17520.png', '18710.png', '19919.png', '23233.png', '06499.png', '03156.png', '19949.png', '10375.png', '04975.png', '18276.png', '00469.png', '10689.png', '16671.png', '07077.png', '09861.png', '14971.png', '14027.png', '09907.png', '06791.png', '22525.png', '20466.png', '03781.png', '01964.png', '17302.png', '02762.png', '10718.png', '07004.png', '15506.png', '01014.png', '18662.png', '20984.png', '21373.png', '19958.png', '04726.png', '07336.png', '22802.png', '09293.png', '01769.png', '07231.png', '10519.png', '20923.png', '02799.png', '17018.png', '07296.png', '01421.png', '22287.png', '14913.png', '01185.png', '09656.png', '06159.png', '22552.png', '21080.png', '06552.png', '12889.png', '19858.png', '19476.png', '04028.png', '10649.png', '16245.png', '18413.png', '08259.png', '18387.png', '19046.png', '07325.png', '08770.png', '06089.png', '17461.png', '14337.png', '00206.png', '16372.png', '06574.png', '24612.png', '10859.png', '06182.png', '20667.png', '17901.png', '20582.png', '11041.png', '08203.png', '18874.png', '19777.png', '10287.png', '16285.png', '19853.png', '20030.png', '23776.png', '22629.png', '13153.png', '15713.png', '15798.png', '09603.png', '20725.png', '10767.png', '16489.png', '12694.png', '12219.png', '13963.png', '10476.png', '02988.png', '17685.png', '16954.png', '06198.png', '03192.png', '11463.png', '08557.png', '24820.png', '06045.png', '20025.png', '08987.png', '04140.png', '06619.png', '07284.png', '15303.png', '16464.png', '14194.png', '08083.png', '23108.png', '00697.png', '13009.png', '01897.png', '00988.png', '01697.png', '13175.png', '22050.png', '23677.png', '22037.png', '06936.png', '07009.png', '06399.png', '01752.png', '05021.png', '13407.png', '24859.png', '10659.png', '04272.png', '03552.png', '11210.png', '20152.png', '17039.png', '13608.png', '24239.png', '18449.png', '20753.png', '05352.png', '16596.png', '13853.png', '05810.png', '17713.png', '16890.png', '04218.png', '12108.png', '04972.png', '12226.png', '18061.png', '24428.png', '18369.png', '24425.png', '08885.png', '13807.png', '21673.png', '23645.png', '14252.png', '19731.png', '13400.png', '16243.png', '08347.png', '24655.png', '06729.png', '21649.png', '22793.png', '00686.png', '00597.png', '23127.png', '21498.png', '17875.png', '05578.png', '03745.png', '23758.png', '18950.png', '05500.png', '10528.png', '18092.png', '11473.png', '02866.png', '20958.png', '06107.png', '20104.png', '06076.png', '15351.png', '08510.png', '10650.png', '17810.png', '19071.png', '13864.png', '20463.png', '14318.png', '10120.png', '11264.png', '08798.png', '09717.png', '08306.png', '18720.png', '09874.png', '14593.png', '10970.png', '15301.png', '04614.png', '11126.png', '22847.png', '11573.png', '16547.png', '08376.png', '00530.png', '04203.png', '17516.png', '08024.png', '16462.png', '20159.png', '15212.png', '01913.png', '17472.png', '11338.png', '02492.png', '00532.png', '05912.png', '21235.png', '15345.png', '21617.png', '03257.png', '00742.png', '24148.png', '07307.png', '06155.png', '06174.png', '23608.png', '04966.png', '12821.png', '22313.png', '24527.png', '10719.png', '15169.png', '13316.png', '16727.png', '14671.png', '22882.png', '23073.png', '22392.png', '16406.png', '24243.png', '20583.png', '24850.png', '07236.png', '08976.png', '17964.png', '08611.png', '07060.png', '22610.png', '11424.png', '22740.png', '02687.png', '07245.png', '10895.png', '02495.png', '03003.png', '20604.png', '23139.png', '22031.png', '18737.png', '22820.png', '14118.png', '19430.png', '10227.png', '21331.png', '17074.png', '08522.png', '16076.png', '06669.png', '02149.png', '15819.png', '17124.png', '23532.png', '00281.png', '21157.png', '22321.png', '05717.png', '11886.png', '00394.png', '00883.png', '18866.png', '12416.png', '09939.png', '21569.png', '08052.png', '04794.png', '18148.png', '19791.png', '24856.png', '11023.png', '01088.png', '22364.png', '20654.png', '23305.png', '08069.png', '14001.png', '11682.png', '02087.png', '24931.png', '14506.png', '02093.png', '23015.png', '02578.png', '07604.png', '20213.png', '08428.png', '12299.png', '16298.png', '20307.png', '02908.png', '08091.png', '00026.png', '21484.png', '05094.png', '22093.png', '12136.png', '04736.png', '01583.png', '14843.png', '06859.png', '13971.png', '08627.png', '20191.png', '12410.png', '17548.png', '14013.png', '03321.png', '17106.png', '23647.png', '03736.png', '17328.png', '10191.png', '23705.png', '23492.png', '11847.png', '11987.png', '17853.png', '15200.png', '21874.png', '01862.png', '09436.png', '17487.png', '20858.png', '02262.png', '07018.png', '02488.png', '06697.png', '01721.png', '18160.png', '01011.png', '06825.png', '01600.png', '24456.png', '23803.png', '19252.png', '24524.png', '06199.png', '10776.png', '13783.png', '12246.png', '10704.png', '21121.png', '09112.png', '14491.png', '18452.png', '09090.png', '21961.png', '18676.png', '04120.png', '20279.png', '23514.png', '10816.png', '17879.png', '17364.png', '01937.png', '21872.png', '11291.png', '23443.png', '22830.png', '14313.png', '01574.png', '21553.png', '00285.png', '13635.png', '07677.png', '21444.png', '04977.png', '19675.png', '07970.png', '06346.png', '11066.png', '08565.png', '00442.png', '06194.png', '08009.png', '11491.png', '23450.png', '24298.png', '14021.png', '08120.png', '07509.png', '12938.png', '00819.png', '05471.png', '05203.png', '05484.png', '24201.png', '16723.png', '12316.png', '19141.png', '10021.png', '07696.png', '13776.png', '01761.png', '18202.png', '18740.png', '22729.png', '05157.png', '02681.png', '16161.png', '03252.png', '03647.png', '24925.png', '03980.png', '23031.png', '20903.png', '16290.png', '14749.png', '20998.png', '14294.png', '14049.png', '08175.png', '24473.png', '00556.png', '05283.png', '07599.png', '06993.png', '14715.png', '19061.png', '16229.png', '01533.png', '21284.png', '13155.png', '03314.png', '04253.png', '08382.png', '12934.png', '17917.png', '03879.png', '20917.png', '12222.png', '22707.png', '22453.png', '13500.png', '06149.png', '09146.png', '07392.png', '04985.png', '12594.png', '03196.png', '10086.png', '13252.png', '00901.png', '21777.png', '19908.png', '18115.png', '19377.png', '05306.png', '03867.png', '02759.png', '10497.png', '18649.png', '03739.png', '01252.png', '04838.png', '07102.png', '03154.png', '12202.png', '14078.png', '14197.png', '04174.png', '06217.png', '18199.png', '01408.png', '20120.png', '13710.png', '07870.png', '19887.png', '18316.png', '02110.png', '23076.png', '08564.png', '08730.png', '15129.png', '24667.png', '13973.png', '18117.png', '20801.png', '21205.png', '12864.png', '21988.png', '16572.png', '21166.png', '21761.png', '23575.png', '24907.png', '20922.png', '07119.png', '08273.png', '09439.png', '14827.png', '01653.png', '12461.png', '10188.png', '05915.png', '23210.png', '10588.png', '03212.png', '10360.png', '10977.png', '18317.png', '10929.png', '04840.png', '12304.png', '22969.png', '14825.png', '09390.png', '23011.png', '00562.png', '07419.png', '24889.png', '21475.png', '16629.png', '07732.png', '15180.png', '05743.png', '07044.png', '17035.png', '23561.png', '12190.png', '22808.png', '00794.png', '14286.png', '23438.png', '01738.png', '19120.png', '09199.png', '15368.png', '09115.png', '15362.png', '03319.png', '02938.png', '17177.png', '11679.png', '16054.png', '04506.png', '07906.png', '16288.png', '09493.png', '12800.png', '14034.png', '10118.png', '04313.png', '23095.png', '12296.png', '05634.png', '19674.png', '13878.png', '09579.png', '18626.png', '22936.png', '21138.png', '03681.png', '07383.png', '10852.png', '08517.png', '24656.png', '02193.png', '16208.png', '06517.png', '12275.png', '18362.png', '21134.png', '24066.png', '05902.png', '01631.png', '01572.png', '00403.png', '08506.png', '18088.png', '06411.png', '22614.png', '09908.png', '13140.png', '17512.png', '01167.png', '23954.png', '16384.png', '06910.png', '15173.png', '24247.png', '24872.png', '15056.png', '07285.png', '05295.png', '22002.png', '10591.png', '16220.png', '18164.png', '18468.png', '02420.png', '12827.png', '13970.png', '16789.png', '23539.png', '15211.png', '19131.png', '08031.png', '20552.png', '06983.png', '20805.png', '19515.png', '18455.png', '21972.png', '18893.png', '20915.png', '01337.png', '18836.png', '09960.png', '14198.png', '07861.png', '17330.png', '18973.png', '12685.png', '03398.png', '24326.png', '04536.png', '10401.png', '09813.png', '04946.png', '19079.png', '17504.png', '18899.png', '10959.png', '19310.png', '22114.png', '19327.png', '03289.png', '19116.png', '12813.png', '07181.png', '00146.png', '19091.png', '07436.png', '03015.png', '16951.png', '01616.png', '03279.png', '18553.png', '15348.png', '19839.png', '18173.png', '10436.png', '12887.png', '09952.png', '24028.png', '00298.png', '14273.png', '08746.png', '16678.png', '20334.png', '19837.png', '18764.png', '03828.png', '13947.png', '01428.png', '22725.png', '13168.png', '24851.png', '18246.png', '23109.png', '23168.png', '04396.png', '18290.png', '10295.png', '13619.png', '05430.png', '09195.png', '19964.png', '18961.png', '02698.png', '24643.png', '00467.png', '03971.png', '14322.png', '20131.png', '08796.png', '05844.png', '22779.png', '15888.png', '00103.png', '18195.png', '12582.png', '23617.png', '03486.png', '03425.png', '14874.png', '09794.png', '12474.png', '01357.png', '13118.png', '01528.png', '13271.png', '08710.png', '01838.png', '19609.png', '00780.png', '24917.png', '06916.png', '18830.png', '01108.png', '21865.png', '24012.png', '23961.png', '16179.png', '23382.png', '07603.png', '19157.png', '21288.png', '04812.png', '00156.png', '04479.png', '19277.png', '08842.png', '23285.png', '20046.png', '19697.png', '05305.png', '16760.png', '18124.png', '08833.png', '09299.png', '15762.png', '12079.png', '12044.png', '01243.png', '12347.png', '09641.png', '14438.png', '22056.png', '00006.png', '06670.png', '15645.png', '06300.png', '19198.png', '09772.png', '21054.png', '00627.png', '01641.png', '04499.png', '00481.png', '11690.png', '12976.png', '00222.png', '02860.png', '09265.png', '23050.png', '00951.png', '08213.png', '15647.png', '03577.png', '18712.png', '23777.png', '24244.png', '01047.png', '12456.png', '08944.png', '12617.png', '17597.png', '02279.png', '11119.png', '01495.png', '14007.png', '09413.png', '04220.png', '02096.png', '06454.png', '13705.png', '15862.png', '01580.png', '10620.png', '11666.png', '16045.png', '00491.png', '02240.png', '21287.png', '19633.png', '13152.png', '21027.png', '22960.png', '18839.png', '03899.png', '04858.png', '13743.png', '16098.png', '08826.png', '00637.png', '08504.png', '11763.png', '15931.png', '03589.png', '14233.png', '08475.png', '10324.png', '06305.png', '24088.png', '19539.png', '13978.png', '11829.png', '18045.png', '19705.png', '24898.png', '15659.png', '01118.png', '24723.png', '21457.png', '05906.png', '21686.png', '19097.png', '00673.png', '10340.png', '07734.png', '13734.png', '19817.png', '19092.png', '10799.png', '09121.png', '00470.png', '08724.png', '09409.png', '08583.png', '14214.png', '00244.png', '12112.png', '02316.png', '11530.png', '15800.png', '08704.png', '16897.png', '08450.png', '18467.png', '21656.png', '12415.png', '15939.png', '03864.png', '08720.png', '13474.png', '04875.png', '08515.png', '17219.png', '07957.png', '05547.png', '09750.png', '12090.png', '04325.png', '22710.png', '02926.png', '06331.png', '09709.png', '03304.png', '16553.png', '22236.png', '16082.png', '15794.png', '24728.png', '04490.png', '16047.png', '00333.png', '19801.png', '07450.png', '23455.png', '02896.png', '20259.png', '11809.png', '02817.png', '04035.png', '13362.png', '08896.png', '16747.png', '04540.png', '21075.png', '17483.png', '19386.png', '24449.png', '02261.png', '09829.png', '14154.png', '04630.png', '15239.png', '22023.png', '07837.png', '03490.png', '17239.png', '00578.png', '22475.png', '10517.png', '11509.png', '10550.png', '05360.png', '12927.png', '14308.png', '13912.png', '21718.png', '14999.png', '01601.png', '09149.png', '06742.png', '22183.png', '10630.png', '14092.png', '23908.png', '17403.png', '13655.png', '09059.png', '16655.png', '13034.png', '20474.png', '08964.png', '06109.png', '01726.png', '10946.png', '21846.png', '10339.png', '01941.png', '15534.png', '11816.png', '07849.png', '24329.png', '22634.png', '08761.png', '17976.png', '16646.png', '02371.png', '02075.png', '21009.png', '04266.png', '20844.png', '04071.png', '03715.png', '06485.png', '21746.png', '14388.png', '06261.png', '11149.png', '24645.png', '17742.png', '04892.png', '07874.png', '06736.png', '09928.png', '21229.png', '24602.png', '11407.png', '05271.png', '19939.png', '04443.png', '07678.png', '24130.png', '04953.png', '02936.png', '18027.png', '16230.png', '03594.png', '06175.png', '07254.png', '03726.png', '16699.png', '00452.png', '09660.png', '04819.png', '03350.png', '03523.png', '16984.png', '20851.png', '21743.png', '17426.png', '03521.png', '10676.png', '15272.png', '04326.png', '15639.png', '02483.png', '05441.png', '04379.png', '20452.png', '16354.png', '05858.png', '21800.png', '01746.png', '03042.png', '22299.png', '10004.png', '14719.png', '07483.png', '18494.png', '14227.png', '24094.png', '07686.png', '13089.png', '17214.png', '02592.png', '16542.png', '05768.png', '13820.png', '02519.png', '10454.png', '17743.png', '06064.png', '23597.png', '00646.png', '18576.png', '11318.png', '21613.png', '12856.png', '05253.png', '01204.png', '02162.png', '23989.png', '18500.png', '13102.png', '15338.png', '22413.png', '04750.png', '09671.png', '10571.png', '00729.png', '24816.png', '05173.png', '05797.png', '21790.png', '13066.png', '11408.png', '10179.png', '09517.png', '14808.png', '22046.png', '07196.png', '06696.png', '18113.png', '21101.png', '20036.png', '24754.png', '16869.png', '19879.png', '14581.png', '10084.png', '21843.png', '16751.png', '16564.png', '20370.png', '09704.png', '21344.png', '21704.png', '11336.png', '14307.png', '01603.png', '01610.png', '15813.png', '12891.png', '09889.png', '06466.png', '04058.png', '15999.png', '04910.png', '16886.png', '10801.png', '02659.png', '19466.png', '06444.png', '00598.png', '12897.png', '08266.png', '19829.png', '11837.png', '22414.png', '10607.png', '18404.png', '05620.png', '24646.png', '17414.png', '06882.png', '02742.png', '11951.png', '15873.png', '22457.png', '23993.png', '03023.png', '23009.png', '07830.png', '13335.png', '02855.png', '18934.png', '14393.png', '21187.png', '18465.png', '08576.png', '17572.png', '21167.png', '08075.png', '22026.png', '23083.png', '17932.png', '12541.png', '02925.png', '10282.png', '00778.png', '21574.png', '08106.png', '07513.png', '24245.png', '09133.png', '02388.png', '10663.png', '08284.png', '22564.png', '08239.png', '00529.png', '20050.png', '11967.png', '00197.png', '12354.png', '12123.png', '24309.png', '09920.png', '13110.png', '17795.png', '02229.png', '08008.png', '22452.png', '22167.png', '09566.png', '24738.png', '15420.png', '01077.png', '01246.png', '23966.png', '21614.png', '17207.png', '03152.png', '08886.png', '24613.png', '03261.png', '03101.png', '00106.png', '09605.png', '20508.png', '18724.png', '20346.png', '01552.png', '13690.png', '23995.png', '14300.png', '03414.png', '07208.png', '05293.png', '01114.png', '20300.png', '07430.png', '00345.png', '19819.png', '04105.png', '01778.png', '13982.png', '16297.png', '01213.png', '07417.png', '01191.png', '17733.png', '04292.png', '21410.png', '24040.png', '19065.png', '21103.png', '19321.png', '23976.png', '07241.png', '16064.png', '12496.png', '11392.png', '06902.png', '20047.png', '14775.png', '00573.png', '24466.png', '06280.png', '18264.png', '14536.png', '21248.png', '24076.png', '16326.png', '06835.png', '13801.png', '22539.png', '14634.png', '01350.png', '18837.png', '12473.png', '03325.png', '13391.png', '12398.png', '12903.png', '02010.png', '19059.png', '16429.png', '22513.png', '04063.png', '08947.png', '15187.png', '17865.png', '11040.png', '15475.png', '08816.png', '09817.png', '16283.png', '00248.png', '11659.png', '12615.png', '11240.png', '17975.png', '01705.png', '02220.png', '08210.png', '11960.png', '21892.png', '19420.png', '05495.png', '17754.png', '07032.png', '13964.png', '03199.png', '02964.png', '07173.png', '04954.png', '12372.png', '20324.png', '23823.png', '16435.png', '21130.png', '02994.png', '10088.png', '13623.png', '04620.png', '14103.png', '04880.png', '08868.png', '12649.png', '17003.png', '18320.png', '07803.png', '15871.png', '20140.png', '01394.png', '05863.png', '10044.png', '08383.png', '14140.png', '04400.png', '20631.png', '04430.png', '06128.png', '11157.png', '20245.png', '20756.png', '14091.png', '24318.png', '05833.png', '06242.png', '07955.png', '11637.png', '11481.png', '05370.png', '22340.png', '00308.png', '19297.png', '12100.png', '14035.png', '10432.png', '04531.png', '24520.png', '00698.png', '05464.png', '15001.png', '24221.png', '08606.png', '19991.png', '10252.png', '02503.png', '02831.png', '02757.png', '12276.png', '05079.png', '22218.png', '14434.png', '12809.png', '08233.png', '21768.png', '12231.png', '18744.png', '07494.png', '08768.png', '07445.png', '20829.png', '01608.png', '13413.png', '10409.png', '03358.png', '02451.png', '23357.png', '01575.png', '02583.png', '06756.png', '09988.png', '05953.png', '12639.png', '12159.png', '11570.png', '02942.png', '14462.png', '13692.png', '14700.png', '09261.png', '24521.png', '05215.png', '13204.png', '09449.png', '14798.png', '22255.png', '05732.png', '03202.png', '16234.png', '06804.png', '15152.png', '07511.png', '10235.png', '15896.png', '11398.png', '16393.png', '07999.png', '09535.png', '11884.png', '10509.png', '20597.png', '20476.png', '12236.png', '23832.png', '10193.png', '22689.png', '13728.png', '21078.png', '19970.png', '03338.png', '05945.png', '03664.png', '04554.png', '21339.png', '14522.png', '08410.png', '21705.png', '19228.png', '13463.png', '19508.png', '14896.png', '01676.png', '21946.png', '10538.png', '05937.png', '00644.png', '15438.png', '10079.png', '05636.png', '22853.png', '09321.png', '03588.png', '02328.png', '01504.png', '09248.png', '22341.png', '12485.png', '10958.png', '16260.png', '18619.png', '12154.png', '12292.png', '10570.png', '11287.png', '07681.png', '17500.png', '01457.png', '23708.png', '20495.png', '23960.png', '03854.png', '18591.png', '06533.png', '05528.png', '15885.png', '05470.png', '24079.png', '09568.png', '00002.png', '01918.png', '20062.png', '23216.png', '23212.png', '08597.png', '17608.png', '24291.png', '23956.png', '22180.png', '08555.png', '02277.png', '00068.png', '17373.png', '11575.png', '13068.png', '18856.png', '23079.png', '03455.png', '03360.png', '10741.png', '11760.png', '09279.png', '13130.png', '14109.png', '04837.png', '20467.png', '23196.png', '18382.png', '20044.png', '19654.png', '03599.png', '24543.png', '14298.png', '23876.png', '18795.png', '04235.png', '00857.png', '16119.png', '02775.png', '02209.png', '12170.png', '17292.png', '07397.png', '22778.png', '11806.png', '12156.png', '06072.png', '15165.png', '13682.png', '20332.png', '02426.png', '09826.png', '10656.png', '06443.png', '06364.png', '01605.png', '00294.png', '17452.png', '22638.png', '06842.png', '00189.png', '03716.png', '00192.png', '19038.png', '21785.png', '11456.png', '24919.png', '23361.png', '06131.png', '21549.png', '02295.png', '13248.png', '21727.png', '13938.png', '09945.png', '10521.png', '08553.png', '20125.png', '07690.png', '16929.png', '14937.png', '20310.png', '09860.png', '24213.png', '19737.png', '20999.png', '07375.png', '17603.png', '14636.png', '07357.png', '00267.png', '13922.png', '18210.png', '09713.png', '10431.png', '05680.png', '02114.png', '24166.png', '16408.png', '01418.png', '10927.png', '03315.png', '01720.png', '00428.png', '20288.png', '04412.png', '18442.png', '09677.png', '05207.png', '19557.png', '04142.png', '23202.png', '09289.png', '11326.png', '13906.png', '01529.png', '15435.png', '20766.png', '15217.png', '23530.png', '19676.png', '11394.png', '16955.png', '16456.png', '10119.png', '12389.png', '07672.png', '10249.png', '09510.png', '06112.png', '16790.png', '24085.png', '23571.png', '12305.png', '13482.png', '22623.png', '13859.png', '05794.png', '09237.png', '08000.png', '15216.png', '05748.png', '16907.png', '11664.png', '11142.png', '01590.png', '22569.png', '17238.png', '23059.png', '21698.png', '09994.png', '07835.png', '20453.png', '11429.png', '16228.png', '02446.png', '13395.png', '20560.png', '03264.png', '23094.png', '24729.png', '24842.png', '08473.png', '09055.png', '08092.png', '17044.png', '20721.png', '13119.png', '04971.png', '05512.png', '20113.png', '20004.png', '12787.png', '12798.png', '18562.png', '02907.png', '00221.png', '01017.png', '09618.png', '16883.png', '11866.png', '05551.png', '06445.png', '20393.png', '12506.png', '21933.png', '10200.png', '01235.png', '07614.png', '11411.png', '06592.png', '23334.png', '01657.png', '18328.png', '08713.png', '17396.png', '09465.png', '03495.png', '15455.png', '08328.png', '02278.png', '09680.png', '07065.png', '05805.png', '21916.png', '23172.png', '06648.png', '10885.png', '17921.png', '07956.png', '15821.png', '06228.png', '06541.png', '04080.png', '09779.png', '09546.png', '00507.png', '09525.png', '13423.png', '01873.png', '20879.png', '04114.png', '03931.png', '24307.png', '06404.png', '24796.png', '07688.png', '19640.png', '17120.png', '02129.png', '17859.png', '03177.png', '09247.png', '05572.png', '05998.png', '13723.png', '09144.png', '20089.png', '19159.png', '02219.png', '11334.png', '24051.png', '20745.png', '07274.png', '06952.png', '24343.png', '17245.png', '04844.png', '21071.png', '12945.png', '00944.png', '14770.png', '24111.png', '04408.png', '06557.png', '00990.png', '14500.png', '23816.png', '22604.png', '01437.png', '09087.png', '06012.png', '15034.png', '07266.png', '04246.png', '04268.png', '11864.png', '03568.png', '09485.png', '03429.png', '02672.png', '13709.png', '18643.png', '07136.png', '20285.png', '22711.png', '02213.png', '06406.png', '00870.png', '06218.png', '15698.png', '15431.png', '02653.png', '24865.png', '18542.png', '08271.png', '05556.png', '16720.png', '20555.png', '24126.png', '12327.png', '21206.png', '23270.png', '05822.png', '16164.png', '17257.png', '23515.png', '16353.png', '17031.png', '19646.png', '08926.png', '22661.png', '04075.png', '16570.png', '22892.png', '17804.png', '08875.png', '22169.png', '09509.png', '11103.png', '16926.png', '16257.png', '10511.png', '23736.png', '06550.png', '11054.png', '22631.png', '24836.png', '17100.png', '08015.png', '15546.png', '17155.png', '04570.png', '11698.png', '09047.png', '08541.png', '12929.png', '12819.png', '18281.png', '18217.png', '12826.png', '11976.png', '07892.png', '18345.png', '09820.png', '06011.png', '24625.png', '19489.png', '10981.png', '04662.png', '15437.png', '07600.png', '20602.png', '12344.png', '05865.png', '10226.png', '22954.png', '23394.png', '00765.png', '19546.png', '19179.png', '04224.png', '15681.png', '08870.png', '22678.png', '04436.png', '10989.png', '03970.png', '20478.png', '12632.png', '14354.png', '06267.png', '01678.png', '01094.png', '22445.png', '20071.png', '09364.png', '08598.png', '13504.png', '24823.png', '23145.png', '22833.png', '14134.png', '03657.png', '09120.png', '09421.png', '22201.png', '13569.png', '06672.png', '18270.png', '11508.png', '24400.png', '21265.png', '21109.png', '23324.png', '10991.png', '16560.png', '02676.png', '09581.png', '14654.png', '24292.png', '18315.png', '06946.png', '07656.png', '23092.png', '16874.png', '10523.png', '08801.png', '04504.png', '18011.png', '03405.png', '10225.png', '12900.png', '04650.png', '09452.png', '02362.png', '03102.png', '14551.png', '14764.png', '04988.png', '16643.png', '14495.png', '21625.png', '16831.png', '15841.png', '17381.png', '21359.png', '01613.png', '13515.png', '08426.png', '23973.png', '05733.png', '09593.png', '14459.png', '18876.png', '05571.png', '01443.png', '00226.png', '19782.png', '01701.png', '20798.png', '03480.png', '02005.png', '14652.png', '22647.png', '05520.png', '21145.png', '09863.png', '22777.png', '15240.png', '06659.png', '01517.png', '04496.png', '12504.png', '19555.png', '19471.png', '00455.png', '07978.png', '14490.png', '15483.png', '19571.png', '20167.png', '17156.png', '05514.png', '18887.png', '02369.png', '12317.png', '22516.png', '06187.png', '03378.png', '07979.png', '10394.png', '08368.png', '18391.png', '08708.png', '09134.png', '01536.png', '13731.png', '11762.png', '00692.png', '08758.png', '13234.png', '23432.png', '00994.png', '07796.png', '01041.png', '21960.png', '20403.png', '24032.png', '04713.png', '24814.png', '11510.png', '13074.png', '11267.png', '00369.png', '13796.png', '14554.png', '16089.png', '24441.png', '03089.png', '23582.png', '04062.png', '12743.png', '04676.png', '01950.png', '03525.png', '11027.png', '14189.png', '18384.png', '24279.png', '17720.png', '19700.png', '10212.png', '08806.png', '17978.png', '02738.png', '02450.png', '10860.png', '24541.png', '16494.png', '05081.png', '05824.png', '17481.png', '05468.png', '05375.png', '19742.png', '15328.png', '09689.png', '00323.png', '10795.png', '02502.png', '23842.png', '19619.png', '09480.png', '17656.png', '14282.png', '20690.png', '18878.png', '04027.png', '07422.png', '14810.png', '19404.png', '04656.png', '04885.png', '10060.png', '08970.png', '06987.png', '02919.png', '21422.png', '23508.png', '09346.png', '05446.png', '08872.png', '15076.png', '11707.png', '08469.png', '24486.png', '09472.png', '05286.png', '10886.png', '02089.png', '02237.png', '07534.png', '24227.png', '24918.png', '23839.png', '10947.png', '10435.png', '06424.png', '21648.png', '04420.png', '19472.png', '18275.png', '07008.png', '11333.png', '12653.png', '21104.png', '18219.png', '12781.png', '18569.png', '15983.png', '12587.png', '12349.png', '22996.png', '08481.png', '04795.png', '22402.png', '23424.png', '24547.png', '15966.png', '05878.png', '12538.png', '17258.png', '14877.png', '04716.png', '04473.png', '17493.png', '15077.png', '11978.png', '22858.png', '11497.png', '07205.png', '17614.png', '09130.png', '21908.png', '15123.png', '24901.png', '14274.png', '16775.png', '11532.png', '16349.png', '02722.png', '02378.png', '24596.png', '15279.png', '07039.png', '03711.png', '12776.png', '16676.png', '16329.png', '06814.png', '17630.png', '08371.png', '13981.png', '16425.png', '00145.png', '24212.png', '12630.png', '00430.png', '00758.png', '02686.png', '15263.png', '16813.png', '22959.png', '20571.png', '15761.png', '18904.png', '11506.png', '21682.png', '24301.png', '04360.png', '02755.png', '21119.png', '24252.png', '17660.png', '01673.png', '01854.png', '08016.png', '00256.png', '00310.png', '09212.png', '24785.png', '08632.png', '10001.png', '08483.png', '02007.png', '20929.png', '06875.png', '16910.png', '22885.png', '11652.png', '22191.png', '03197.png', '21589.png', '14365.png', '11980.png', '20612.png', '24253.png', '04434.png', '20773.png', '12623.png', '16884.png', '01105.png', '13092.png', '19003.png', '15849.png', '08975.png', '06991.png', '14425.png', '21085.png', '01685.png', '13027.png', '18655.png', '21086.png', '24489.png', '08268.png', '23807.png', '11277.png', '16190.png', '03763.png', '10537.png', '03209.png', '15721.png', '00862.png', '09747.png', '16824.png', '24892.png', '08703.png', '02600.png', '22267.png', '02703.png', '07207.png', '02869.png', '21744.png', '07992.png', '01615.png', '15617.png', '21501.png', '06833.png', '24874.png', '04757.png', '20374.png', '04011.png', '06495.png', '20590.png', '14797.png', '19580.png', '23754.png', '20263.png', '19940.png', '14455.png', '02207.png', '09927.png', '03354.png', '06197.png', '07082.png', '08694.png', '04637.png', '02140.png', '17805.png', '23418.png', '12989.png', '17404.png', '10341.png', '05795.png', '18212.png', '19526.png', '07709.png', '05466.png', '01480.png', '02587.png', '22871.png', '21669.png', '04076.png', '18789.png', '13049.png', '21303.png', '05288.png', '08100.png', '18077.png', '21662.png', '23683.png', '10403.png', '22315.png', '24471.png', '17169.png', '11004.png', '11956.png', '01966.png', '03058.png', '07733.png', '16244.png', '16627.png', '10199.png', '10378.png', '01881.png', '15185.png', '23101.png', '06808.png', '00823.png', '07269.png', '07059.png', '00080.png', '08575.png', '18304.png', '15847.png', '15363.png', '20123.png', '20685.png', '21771.png', '06034.png', '00171.png', '24770.png', '09286.png', '07934.png', '03435.png', '12568.png', '09381.png', '13834.png', '17788.png', '00957.png', '18956.png', '20407.png', '00972.png', '00357.png', '17538.png', '05247.png', '10830.png', '07341.png', '17529.png', '14181.png', '02647.png', '07066.png', '03550.png', '19765.png', '04593.png', '04386.png', '10209.png', '02181.png', '22488.png', '02656.png', '09784.png', '20265.png', '17571.png', '05955.png', '21094.png', '01992.png', '14155.png', '23067.png', '24536.png', '08544.png', '24800.png', '20460.png', '23413.png', '24372.png', '01962.png', '24468.png', '19474.png', '18890.png', '02047.png', '14317.png', '11363.png', '03909.png', '16637.png', '17172.png', '09123.png', '05273.png', '07985.png', '08973.png', '24860.png', '18753.png', '21403.png', '20676.png', '10010.png', '13850.png', '11814.png', '06599.png', '19810.png', '00682.png', '20049.png', '22760.png', '02349.png', '15989.png', '19966.png', '02559.png', '03773.png', '04986.png', '10768.png', '07904.png', '00487.png', '23282.png', '15936.png', '07390.png', '16252.png', '21271.png', '16431.png', '03497.png', '13918.png', '23593.png', '16838.png', '15288.png', '19259.png', '20154.png', '23021.png', '16920.png', '03004.png', '21755.png', '01861.png', '06301.png', '18434.png', '05616.png', '13881.png', '11915.png', '13368.png', '04001.png', '05190.png', '13951.png', '04383.png', '05498.png', '01279.png', '09335.png', '07821.png', '06252.png', '08470.png', '03795.png', '08302.png', '20011.png', '15663.png', '21635.png', '23863.png', '00131.png', '23905.png', '11182.png', '01991.png', '18653.png', '00161.png', '18760.png', '20718.png', '04648.png', '17015.png', '17501.png', '16687.png', '23655.png', '20778.png', '01863.png', '20813.png', '11347.png', '18054.png', '03193.png', '19517.png', '18969.png', '15957.png', '23286.png', '15649.png', '24590.png', '15287.png', '01625.png', '18757.png', '22112.png', '18014.png', '16467.png', '03168.png', '12014.png', '19530.png', '02217.png', '13305.png', '07726.png', '14904.png', '12527.png', '20715.png', '01434.png', '01297.png', '14450.png', '00570.png', '21191.png', '17296.png', '03403.png', '03428.png', '06156.png', '17247.png', '22222.png', '18558.png', '16392.png', '01306.png', '19728.png', '13301.png', '02281.png', '04645.png', '15976.png', '13467.png', '01908.png', '23744.png', '07094.png', '02798.png', '17278.png', '16873.png', '21738.png', '12206.png', '09235.png', '20657.png', '00318.png', '02023.png', '22524.png', '01890.png', '06940.png', '10423.png', '16822.png', '12139.png', '07682.png', '20579.png', '22069.png', '02423.png', '01634.png', '05890.png', '18266.png', '15132.png', '02822.png', '15815.png', '21155.png', '21018.png', '07550.png', '15850.png', '09731.png', '10052.png', '15789.png', '19338.png', '06050.png', '10114.png', '13910.png', '01137.png', '08884.png', '01700.png', '11153.png', '19441.png', '13353.png', '14386.png', '24150.png', '15619.png', '18492.png', '05105.png', '20600.png', '15908.png', '01155.png', '11193.png', '05217.png', '01144.png', '18850.png', '15156.png', '03743.png', '13590.png', '07328.png', '05876.png', '16066.png', '17560.png', '01129.png', '10585.png', '20632.png', '11204.png', '08460.png', '13136.png', '14757.png', '02286.png', '10996.png', '22758.png', '14385.png', '06698.png', '22104.png', '14932.png', '01977.png', '17318.png', '19578.png', '01581.png', '00073.png', '03788.png', '05720.png', '23194.png', '02675.png', '09810.png', '18685.png', '23516.png', '16332.png', '01824.png', '21182.png', '14949.png', '13257.png', '00515.png', '09092.png', '24410.png', '20066.png', '00174.png', '14484.png', '18179.png', '01065.png', '18383.png', '22872.png', '16004.png', '21739.png', '18012.png', '13290.png', '00805.png', '11511.png', '11810.png', '14958.png', '04961.png', '16158.png', '11483.png', '02414.png', '23123.png', '17841.png', '03969.png', '10214.png', '13061.png', '08772.png', '24049.png', '22328.png', '16318.png', '09830.png', '12256.png', '05251.png', '18678.png', '19218.png', '20705.png', '20281.png', '20814.png', '08756.png', '15802.png', '10334.png', '01898.png', '11495.png', '14213.png', '10504.png', '11592.png', '11689.png', '06071.png', '09645.png', '12673.png', '11486.png', '11049.png', '00233.png', '06190.png', '11677.png', '23420.png', '10587.png', '15196.png', '00315.png', '21668.png', '18766.png', '21305.png', '16135.png', '20587.png', '00880.png', '08487.png', '12931.png', '13433.png', '12829.png', '23936.png', '18702.png', '24369.png', '04651.png', '05947.png', '14572.png', '11861.png', '17983.png', '15845.png', '18867.png', '04690.png', '06615.png', '16765.png', '02549.png', '12519.png', '07582.png', '16432.png', '14280.png', '19834.png', '05605.png', '07159.png', '23200.png', '17739.png', '24689.png', '00432.png', '17385.png', '14523.png', '22773.png', '23828.png', '24068.png', '08534.png', '15460.png', '00583.png', '15725.png', '01355.png', '20143.png', '00512.png', '00126.png', '05210.png', '23307.png', '19351.png', '20357.png', '16784.png', '19494.png', '13985.png', '18324.png', '14778.png', '17148.png', '06394.png', '14086.png', '09116.png', '18577.png', '20852.png', '04188.png', '05809.png', '14209.png', '18463.png', '02139.png', '05701.png', '13112.png', '22540.png', '03469.png', '08396.png', '04873.png', '20992.png', '17679.png', '09228.png', '12297.png', '09487.png', '17054.png', '08464.png', '06911.png', '11757.png', '23316.png', '23743.png', '18133.png', '15894.png', '09775.png', '08141.png', '12272.png', '09280.png', '18721.png', '19005.png', '22410.png', '02285.png', '08194.png', '23755.png', '22117.png', '24749.png', '02039.png', '03908.png', '10333.png', '09690.png', '24120.png', '15184.png', '12147.png', '20997.png', '02795.png', '03651.png', '11717.png', '22261.png', '17554.png', '06564.png', '11164.png', '18438.png', '08218.png', '10825.png', '04861.png', '07282.png', '10930.png', '14344.png', '00373.png', '13384.png', '22038.png', '08183.png', '07983.png', '24261.png', '06943.png', '01402.png', '01818.png', '18399.png', '13076.png', '20180.png', '23849.png', '10701.png', '15067.png', '23866.png', '19832.png', '14047.png', '00262.png', '11759.png', '01035.png', '16357.png', '18527.png', '09869.png', '06861.png', '12717.png', '23749.png', '17336.png', '04613.png', '16017.png', '00128.png', '00201.png', '15656.png', '10386.png', '13571.png', '16394.png', '00776.png', '02143.png', '22556.png', '02694.png', '02080.png', '20377.png', '05338.png', '16622.png', '22113.png', '12572.png', '03798.png', '06577.png', '08702.png', '13324.png', '12339.png', '14724.png', '08784.png', '15021.png', '03506.png', '06822.png', '18236.png', '11974.png', '00659.png', '18985.png', '10306.png', '17123.png', '05264.png', '08871.png', '22423.png', '08711.png', '08949.png', '18707.png', '04104.png', '14079.png', '02204.png', '13798.png', '24620.png', '02424.png', '03022.png', '05281.png', '00808.png', '08655.png', '12369.png', '22751.png', '11311.png', '15828.png', '11389.png', '23805.png', '09965.png', '12262.png', '21424.png', '05803.png', '08817.png', '17329.png', '12973.png', '00772.png', '12467.png', '24965.png', '18400.png', '15215.png', '14889.png', '11852.png', '20991.png', '17277.png', '03494.png', '20672.png', '21058.png', '17522.png', '06975.png', '01190.png', '08519.png', '07908.png', '21753.png', '09254.png', '14301.png', '05802.png', '05996.png', '08104.png', '24313.png', '14374.png', '09153.png', '16566.png', '11110.png', '22805.png', '23098.png', '07697.png', '18055.png', '12361.png', '12728.png', '20282.png', '18004.png', '17435.png', '00681.png', '08953.png', '11641.png', '11364.png', '17836.png', '24096.png', '09891.png', '17577.png', '11089.png', '05308.png', '11063.png', '20553.png', '13128.png', '00762.png', '07758.png', '16501.png', '16754.png', '08919.png', '03737.png', '19213.png', '08455.png', '23953.png', '14518.png', '13780.png', '09004.png', '15527.png', '22786.png', '11479.png', '10099.png', '21044.png', '03950.png', '03839.png', '21033.png', '16420.png', '03213.png', '07034.png', '01263.png', '14314.png', '01715.png', '17909.png', '20354.png', '15690.png', '23934.png', '05076.png', '13198.png', '21536.png', '22735.png', '17220.png', '01198.png', '06419.png', '13535.png', '02256.png', '01669.png', '18488.png', '05633.png', '13790.png', '24890.png', '00336.png', '10419.png', '01982.png', '13609.png', '10228.png', '07653.png', '07829.png', '02051.png', '11567.png', '06540.png', '21564.png', '14625.png', '18104.png', '01883.png', '15275.png', '05877.png', '24157.png', '01036.png', '07191.png', '21304.png', '03699.png', '14537.png', '20671.png', '14884.png', '00053.png', '24027.png', '13239.png', '05384.png', '18886.png', '12654.png', '21646.png', '02404.png', '11901.png', '12143.png', '19607.png', '00864.png', '11788.png', '02171.png', '15743.png', '17655.png', '17545.png', '24866.png', '21163.png', '10716.png', '03323.png', '17556.png', '03926.png', '11681.png', '07945.png', '24690.png', '01757.png', '21013.png', '17899.png', '03796.png', '19390.png', '07074.png', '13543.png', '19906.png', '03431.png', '22982.png', '13522.png', '06025.png', '13819.png', '01200.png', '04338.png', '08823.png', '10046.png', '21437.png', '24430.png', '15207.png', '10393.png', '09832.png', '12666.png', '01335.png', '20628.png', '01115.png', '16621.png', '13026.png', '11518.png', '05708.png', '03848.png', '20308.png', '11603.png', '19087.png', '14037.png', '23578.png', '22219.png', '21598.png', '24891.png', '18444.png', '02064.png', '11964.png', '13576.png', '17730.png', '06792.png', '01438.png', '21537.png', '02950.png', '21270.png', '05807.png', '14898.png', '23671.png', '06249.png', '11490.png', '23939.png', '15144.png', '22134.png', '12947.png', '17902.png', '23288.png', '15866.png', '20599.png', '12684.png', '09051.png', '05757.png', '12406.png', '05698.png', '16096.png', '15321.png', '09877.png', '14095.png', '16807.png', '05183.png', '13572.png', '13838.png', '00392.png', '09324.png', '13717.png', '10028.png', '17552.png', '21722.png', '08766.png', '24011.png', '20890.png', '13967.png', '10937.png', '12955.png', '16815.png', '24438.png', '12280.png', '01374.png', '16311.png', '20856.png', '21856.png', '13375.png', '23052.png', '20698.png', '23894.png', '02127.png', '12878.png', '11470.png', '13088.png', '09749.png', '06502.png', '20043.png', '04459.png', '04123.png', '23389.png', '21447.png', '12133.png', '21925.png', '07507.png', '18671.png', '11683.png', '17345.png', '09648.png', '01149.png', '17405.png', '13127.png', '19583.png', '15618.png', '02956.png', '12831.png', '10684.png', '17223.png', '18260.png', '07350.png', '21384.png', '05143.png', '09129.png', '14398.png', '19314.png', '21122.png', '09518.png', '08102.png', '08533.png', '06584.png', '08532.png', '16378.png', '17281.png', '09011.png', '00168.png', '19085.png', '10828.png', '01981.png', '21546.png', '06088.png', '21610.png', '23820.png', '03865.png', '12507.png', '13735.png', '20875.png', '12753.png', '20815.png', '05198.png', '14029.png', '17477.png', '15099.png', '00065.png', '00613.png', '00992.png', '24119.png', '00911.png', '19353.png', '09847.png', '20161.png', '09978.png', '10687.png', '03968.png', '22681.png', '08030.png', '21896.png', '22235.png', '20336.png', '11632.png', '21400.png', '07804.png', '19944.png', '18454.png', '04462.png', '24626.png', '08797.png', '00036.png', '00580.png', '09226.png', '13874.png', '12607.png', '14476.png', '18023.png', '22581.png', '11232.png', '14594.png', '09598.png', '04425.png', '13630.png', '06480.png', '13753.png', '13256.png', '22553.png', '18356.png', '15036.png', '18306.png', '15214.png', '12057.png', '15923.png', '04284.png', '21499.png', '00807.png', '18952.png', '23400.png', '15558.png', '02617.png', '05202.png', '13479.png', '09601.png', '24052.png', '17455.png', '12589.png', '08206.png', '24058.png', '16557.png', '12075.png', '10632.png', '09508.png', '04437.png', '09751.png', '11108.png', '05554.png', '02363.png', '20322.png', '04033.png', '06704.png', '22297.png', '24954.png', '00950.png', '18278.png', '11446.png', '00518.png', '03632.png', '07309.png', '23293.png', '24772.png', '01762.png', '21452.png', '08898.png', '23372.png', '02740.png', '20059.png', '10714.png', '14217.png', '01358.png', '00122.png', '09197.png', '22990.png', '19490.png', '24132.png', '21838.png', '10968.png', '02401.png', '18511.png', '07105.png', '04525.png', '03471.png', '16193.png', '11623.png', '05478.png', '09766.png', '09428.png', '21421.png', '18338.png', '11676.png', '22244.png', '03233.png', '23783.png', '08387.png', '11936.png', '08959.png', '05224.png', '04057.png', '21055.png', '08980.png', '10761.png', '19053.png', '19808.png', '21851.png', '13817.png', '06998.png', '17671.png', '13577.png', '20658.png', '08305.png', '17565.png', '22372.png', '11270.png', '16641.png', '11132.png', '09416.png', '16825.png', '01419.png', '17699.png', '20513.png', '23207.png', '00690.png', '05594.png', '19122.png', '07885.png', '19979.png', '01003.png', '12974.png', '04878.png', '01244.png', '06590.png', '01648.png', '21971.png', '04077.png', '11874.png', '02029.png', '22617.png', '11911.png', '11457.png', '02003.png', '18130.png', '01817.png', '01965.png', '13294.png', '08777.png', '21336.png', '12458.png', '12134.png', '18769.png', '08157.png', '02665.png', '06713.png', '16077.png', '20529.png', '07200.png', '16714.png', '17077.png', '08807.png', '14275.png', '13499.png', '20849.png', '08864.png', '09417.png', '20276.png', '06471.png', '22107.png', '12915.png', '23268.png', '00642.png', '07619.png', '15110.png', '12445.png', '24621.png', '10875.png', '03605.png', '12906.png', '15087.png', '21570.png', '23260.png', '11855.png', '22583.png', '17016.png', '14973.png', '21965.png', '16078.png', '02806.png', '17782.png', '21870.png', '17058.png', '01286.png', '00164.png', '07873.png', '11668.png', '01116.png', '20611.png', '00790.png', '24948.png', '04463.png', '23542.png', '13546.png', '03143.png', '16097.png', '22109.png', '12173.png', '00555.png', '08049.png', '04772.png', '14442.png', '03142.png', '10066.png', '19108.png', '08096.png', '01048.png', '08754.png', '02299.png', '09471.png', '03448.png', '11224.png', '04144.png', '21873.png', '17589.png', '21803.png', '11611.png', '00981.png', '17813.png', '04769.png', '02970.png', '04696.png', '07488.png', '17476.png', '15975.png', '21886.png', '22248.png', '14704.png', '20084.png', '04642.png', '12696.png', '19664.png', '23141.png', '13025.png', '02728.png', '08603.png', '24654.png', '19577.png', '16140.png', '15833.png', '08789.png', '13843.png', '05191.png', '08456.png', '12638.png', '11070.png', '15837.png', '12464.png', '02916.png', '03712.png', '20207.png', '21720.png', '11422.png', '07714.png', '21364.png', '07788.png', '13033.png', '03676.png', '02178.png', '02863.png', '13799.png', '11324.png', '03891.png', '01412.png', '03357.png', '02058.png', '15935.png', '14246.png', '09957.png', '17012.png', '23018.png', '16320.png', '23255.png', '08692.png', '04526.png', '13758.png', '03271.png', '16563.png', '24139.png', '05218.png', '03355.png', '16389.png', '03961.png', '11169.png', '24116.png', '06576.png', '10643.png', '20009.png', '10266.png', '14104.png', '13942.png', '09688.png', '03718.png', '11851.png', '12958.png', '18433.png', '03001.png', '06593.png', '23221.png', '13987.png', '24459.png', '02046.png', '16616.png', '21670.png', '01618.png', '00474.png', '12857.png', '15694.png', '02517.png', '02265.png', '02504.png', '22943.png', '08219.png', '11114.png', '01315.png', '02876.png', '23457.png', '03327.png', '10562.png', '21819.png', '15466.png', '16718.png', '21992.png', '22976.png', '00238.png', '14942.png', '12510.png', '13363.png', '22731.png', '18005.png', '21702.png', '15565.png', '16879.png', '13764.png', '21706.png', '00958.png', '01706.png', '17126.png', '10217.png', '15037.png', '17757.png', '17001.png', '23784.png', '09600.png', '12047.png', '14962.png', '04754.png', '16826.png', '13722.png', '13304.png', '05485.png', '17953.png', '13948.png', '24293.png', '00753.png', '16020.png', '07015.png', '17075.png', '16361.png', '09323.png', '21998.png', '17723.png', '10572.png', '20289.png', '10163.png', '16487.png', '13180.png', '13203.png', '12338.png', '04340.png', '24233.png', '04026.png', '01388.png', '00784.png', '05837.png', '13249.png', '04127.png', '04938.png', '14230.png', '18525.png', '11933.png', '21479.png', '06196.png', '08288.png', '09030.png', '05028.png', '04895.png', '23526.png', '09155.png', '06780.png', '08463.png', '24205.png', '19285.png', '00842.png', '21685.png', '13498.png', '11475.png', '15004.png', '15273.png', '13004.png', '24805.png', '24373.png', '01946.png', '15188.png', '20872.png', '10711.png', '15592.png', '12462.png', '23066.png', '15664.png', '12505.png', '11439.png', '08984.png', '01807.png', '01667.png', '20353.png', '05831.png', '14356.png', '14560.png', '18518.png', '02022.png', '00678.png', '06897.png', '09085.png', '12545.png', '22806.png', '17615.png', '01353.png', '09575.png', '11109.png', '21576.png', '03067.png', '04411.png', '13704.png', '08867.png', '23507.png', '05155.png', '02460.png', '15717.png', '06324.png', '14728.png', '01084.png', '17019.png', '12706.png', '21726.png', '13558.png', '18697.png', '16206.png', '11186.png', '12229.png', '20960.png', '11343.png', '01784.png', '17473.png', '19317.png', '08435.png', '03538.png', '05051.png', '16167.png', '15567.png', '14464.png', '16003.png', '06059.png', '08512.png', '22393.png', '16573.png', '15677.png', '20510.png', '08111.png', '16576.png', '05815.png', '23224.png', '24361.png', '17071.png', '24193.png', '18390.png', '12130.png', '14705.png', '09803.png', '13493.png', '02815.png', '11312.png', '13053.png', '13943.png', '00022.png', '06544.png', '10098.png', '19235.png', '06661.png', '07927.png', '21936.png', '00635.png', '19924.png', '22229.png', '04555.png', '13358.png', '14268.png', '22308.png', '22668.png', '06812.png', '16773.png', '07343.png', '15968.png', '13351.png', '15319.png', '04465.png', '14680.png', '12028.png', '16673.png', '00368.png', '23589.png', '16507.png', '02459.png', '21136.png', '08440.png', '17818.png', '00639.png', '02383.png', '00247.png', '20591.png', '01747.png', '24161.png', '18777.png', '19645.png', '05111.png', '09910.png', '08171.png', '09399.png', '11792.png', '00407.png', '06513.png', '08785.png', '01716.png', '00695.png', '00947.png', '14562.png', '21237.png', '01272.png', '14769.png', '09506.png', '07702.png', '16057.png', '04979.png', '05854.png', '00266.png', '23978.png', '02049.png', '10139.png', '11269.png', '04299.png', '11096.png', '09513.png', '03764.png', '10185.png', '09443.png', '20500.png', '23800.png', '08689.png', '12898.png', '21306.png', '00405.png', '00938.png', '21735.png', '09773.png', '09602.png', '04749.png', '03091.png', '03475.png', '09881.png', '04682.png', '14982.png', '09915.png', '18987.png', '20939.png', '06957.png', '23791.png', '19318.png', '19006.png', '24284.png', '12289.png', '24046.png', '16523.png', '10691.png', '09815.png', '09683.png', '02920.png', '09667.png', '04315.png', '04673.png', '21487.png', '23096.png', '19160.png', '03836.png', '23592.png', '02086.png', '00962.png', '23840.png', '12341.png', '17236.png', '01277.png', '19391.png', '01138.png', '19691.png', '09956.png', '12408.png', '15075.png', '02673.png', '14511.png', '17309.png', '03422.png', '08176.png', '15581.png', '08602.png', '13485.png', '03254.png', '08566.png', '17634.png', '23799.png', '03913.png', '20951.png', '03720.png', '06623.png', '17986.png', '13600.png', '13873.png', '09223.png', '22577.png', '01139.png', '19345.png', '17255.png', '20898.png', '20702.png', '07489.png', '16834.png', '20959.png', '01278.png', '02889.png', '21784.png', '10542.png', '05859.png', '11644.png', '24050.png', '19316.png', '16477.png', '24959.png', '12140.png', '03294.png', '18445.png', '12622.png', '14852.png', '11152.png', '17903.png', '13091.png', '02767.png', '19749.png', '08364.png', '00454.png', '16153.png', '19892.png', '21884.png', '22641.png', '14945.png', '12690.png', '10077.png', '04486.png', '14353.png', '18424.png', '09277.png', '20309.png', '12983.png', '14191.png', '21573.png', '17574.png', '09740.png', '03797.png', '18331.png', '19060.png', '11591.png', '09659.png', '07314.png', '10686.png', '01795.png', '07567.png', '23261.png', '03381.png', '07374.png', '12053.png', '06745.png', '16729.png', '18822.png', '17985.png', '04438.png', '00451.png', '23383.png', '01166.png', '18559.png', '21877.png', '03383.png', '06356.png', '03498.png', '14651.png', '20884.png', '05491.png', '16761.png', '03503.png', '03130.png', '05628.png', '09337.png', '24490.png', '13189.png', '18624.png', '21172.png', '22447.png', '19776.png', '14513.png', '19636.png', '08167.png', '01994.png', '01386.png', '21562.png', '12180.png', '20221.png', '24192.png', '06010.png', '21255.png', '23759.png', '20845.png', '18940.png', '18040.png', '06022.png', '03639.png', '23543.png', '03039.png', '03396.png', '00803.png', '02872.png', '00519.png', '05074.png', '06225.png', '22251.png', '05181.png', '03020.png', '17401.png', '19477.png', '23253.png', '08514.png', '21666.png', '07162.png', '03420.png', '12746.png', '13410.png', '12328.png', '00440.png', '20450.png', '17320.png', '16249.png', '14529.png', '24341.png', '18983.png', '22496.png', '05024.png', '12423.png', '13065.png', '11731.png', '04864.png', '21099.png', '02917.png', '11381.png', '17458.png', '01931.png', '23410.png', '08181.png', '03553.png', '14279.png', '02115.png', '12784.png', '11774.png', '22971.png', '04440.png', '23462.png', '17544.png', '05052.png', '22239.png', '14463.png', '16111.png', '18719.png', '24048.png', '20988.png', '08326.png', '18163.png', '12682.png', '15529.png', '21596.png', '00592.png', '15352.png', '16388.png', '05303.png', '20083.png', '05592.png', '13616.png', '01837.png', '09633.png', '23104.png', '09424.png', '23681.png', '19190.png', '19818.png', '05860.png', '14349.png', '06684.png', '20037.png', '01403.png', '06373.png', '03394.png', '14672.png', '13326.png', '10269.png', '04587.png', '17984.png', '22195.png', '17246.png', '17693.png', '02820.png', '13003.png', '18645.png', '16960.png', '20859.png', '09434.png', '00142.png', '18578.png', '12560.png', '11580.png', '21228.png', '00608.png', '08466.png', '06837.png', '02396.png', '22111.png', '00374.png', '02091.png', '12718.png', '18955.png', '04996.png', '19371.png', '08567.png', '16581.png', '03361.png', '14575.png', '16413.png', '05263.png', '13681.png', '06786.png', '02045.png', '21509.png', '20199.png', '22926.png', '12443.png', '02884.png', '23949.png', '17095.png', '24101.png', '24200.png', '10926.png', '05746.png', '07462.png', '24493.png', '07021.png', '04279.png', '13470.png', '14081.png', '21233.png', '22796.png', '04295.png', '01514.png', '11586.png', '20138.png', '13451.png', '17341.png', '16085.png', '14564.png', '04121.png', '06420.png', '02268.png', '12116.png', '24871.png', '18273.png', '06405.png', '21898.png', '08616.png', '00900.png', '02535.png', '03723.png', '11171.png', '07344.png', '05070.png', '08673.png', '16401.png', '10237.png', '03393.png', '12513.png', '19948.png', '23707.png', '16638.png', '01103.png', '14846.png', '15848.png', '19826.png', '02658.png', '17390.png', '11396.png', '02679.png', '20205.png', '19224.png', '16105.png', '16745.png', '17758.png', '08620.png', '22834.png', '06596.png', '00980.png', '07085.png', '10144.png', '01639.png', '23458.png', '15450.png', '22907.png', '07322.png', '03059.png', '03992.png', '11900.png', '10806.png', '10268.png', '12033.png', '07676.png', '18397.png', '11214.png', '11947.png', '18474.png', '07359.png', '14565.png', '12825.png', '10522.png', '20029.png', '17313.png', '03229.png', '10387.png', '14526.png', '09294.png', '07063.png', '05581.png', '08353.png', '23510.png', '07150.png', '02131.png', '11647.png', '00982.png', '10222.png', '04922.png', '11875.png', '10169.png', '17698.png', '21560.png', '03834.png', '17089.png', '24070.png', '04859.png', '15242.png', '12532.png', '17348.png', '05707.png', '16623.png', '10430.png', '05691.png', '03175.png', '06677.png', '14504.png', '08572.png', '09100.png', '05891.png', '07871.png', '12643.png', '13412.png', '01375.png', '18288.png', '08794.png', '10132.png', '23417.png', '10867.png', '21218.png', '12849.png', '07406.png', '17259.png', '17216.png', '10238.png', '24454.png', '09431.png', '18459.png', '16450.png', '14422.png', '13371.png', '05005.png', '07806.png', '04846.png', '16281.png', '18566.png', '05699.png', '22396.png', '09729.png', '24519.png', '19711.png', '21548.png', '07154.png', '15370.png', '20752.png', '15710.png', '19178.png', '13378.png', '10936.png', '16423.png', '01860.png', '15447.png', '19070.png', '08782.png', '02782.png', '08448.png', '20172.png', '13417.png', '21169.png', '18735.png', '01401.png', '04039.png', '13996.png', '15973.png', '09855.png', '21525.png', '06314.png', '23235.png', '24881.png', '14951.png', '20630.png', '02903.png', '16481.png', '01173.png', '24923.png', '23797.png', '04729.png', '14872.png', '12837.png', '04698.png', '11571.png', '00881.png', '07219.png', '05029.png', '19683.png', '09012.png', '20451.png', '07897.png', '09878.png', '15899.png', '12069.png', '23629.png', '24452.png', '05818.png', '20797.png', '21661.png', '17508.png', '22510.png', '07472.png', '18065.png', '17176.png', '12644.png', '19158.png', '22878.png', '16347.png', '15750.png', '07535.png', '22886.png', '18227.png', '18425.png', '24275.png', '09397.png', '01798.png', '13369.png', '09530.png', '18937.png', '23559.png', '20174.png', '14117.png', '18482.png', '14325.png', '22172.png', '06711.png', '09972.png', '01121.png', '03301.png', '10835.png', '10548.png', '14922.png', '11747.png', '24829.png', '12577.png', '04143.png', '04574.png', '05509.png', '13588.png', '04701.png', '12179.png', '14545.png', '08971.png', '15636.png', '16034.png', '07531.png', '01466.png', '19863.png', '14153.png', '09944.png', '06489.png', '01602.png', '10842.png', '06768.png', '13178.png', '18691.png', '08482.png', '13050.png', '00588.png', '09721.png', '20726.png', '18687.png', '22567.png', '14582.png', '13563.png', '03615.png', '01004.png', '15913.png', '16470.png', '15657.png', '07633.png', '14549.png', '24616.png', '22492.png', '17359.png', '18516.png', '22441.png', '22734.png', '23039.png', '20919.png', '01446.png', '09278.png', '17846.png', '24629.png', '24208.png', '05932.png', '20840.png', '14216.png', '01091.png', '16601.png', '09411.png', '23160.png', '17643.png', '02563.png', '19629.png', '20229.png', '11970.png', '18144.png', '17905.png', '05835.png', '13701.png', '15650.png', '11043.png', '12668.png', '14131.png', '19465.png', '16302.png', '05981.png', '01142.png', '01426.png', '21588.png', '06821.png', '19309.png', '06722.png', '24460.png', '18032.png', '14150.png', '05660.png', '03051.png', '05780.png', '07703.png', '14102.png', '14615.png', '11140.png', '19800.png', '02428.png', '11068.png', '13586.png', '01233.png', '17762.png', '21256.png', '13711.png', '00029.png', '21660.png', '04090.png', '05139.png', '12216.png', '19182.png', '10738.png', '16946.png', '03166.png', '08103.png', '22217.png', '19256.png', '18150.png', '08110.png', '03597.png', '19275.png', '04928.png', '17943.png', '03018.png', '07591.png', '09219.png', '17610.png', '03108.png', '14165.png', '10858.png', '04228.png', '13106.png', '20425.png', '07070.png', '00889.png', '18166.png', '04763.png', '23619.png', '01160.png', '02588.png', '16850.png', '15041.png', '00108.png', '15182.png', '18549.png', '19617.png', '01833.png', '16630.png', '13521.png', '16726.png', '01161.png', '12674.png', '08006.png', '12853.png', '13475.png', '08307.png', '03928.png', '03766.png', '18374.png', '12144.png', '05156.png', '18294.png', '13994.png', '00832.png', '10194.png', '09578.png', '03334.png', '03400.png', '14152.png', '19379.png', '05655.png', '10019.png', '13438.png', '18440.png', '10302.png', '16930.png', '08878.png', '06638.png', '18155.png', '22970.png', '22521.png', '23150.png', '07435.png', '11873.png', '00020.png', '20409.png', '07458.png', '05705.png', '11050.png', '17914.png', '19760.png', '04899.png', '02072.png', '09068.png', '18313.png', '11101.png', '19143.png', '09317.png', '22918.png', '17860.png', '20642.png', '06383.png', '24358.png', '01184.png', '17234.png', '11803.png', '09900.png', '08946.png', '16952.png', '00348.png', '08087.png', '09124.png', '00734.png', '17692.png', '09573.png', '11055.png', '18375.png', '23970.png', '23317.png', '11934.png', '06968.png', '22202.png', '20613.png', '03673.png', '03524.png', '12688.png', '07270.png', '23982.png', '10848.png', '13724.png', '15511.png', '03772.png', '05905.png', '21220.png', '05558.png', '12284.png', '08478.png', '00119.png', '12172.png', '04766.png', '09754.png', '16092.png', '11181.png', '12626.png', '21741.png', '04217.png', '00523.png', '08135.png', '04257.png', '14950.png', '13448.png', '11973.png', '09577.png', '15496.png', '08029.png', '17877.png', '01733.png', '18718.png', '06758.png', '10938.png', '18609.png', '16939.png', '21261.png', '23125.png', '09796.png', '07777.png', '00732.png', '11465.png', '15145.png', '01538.png', '01361.png', '17052.png', '08086.png', '20974.png', '08125.png', '13905.png', '19994.png', '17392.png', '05341.png', '08359.png', '07771.png', '06378.png', '16660.png', '09284.png', '22437.png', '04343.png', '23294.png', '23488.png', '23685.png', '24572.png', '00416.png', '04668.png', '02750.png', '11275.png', '18543.png', '19896.png', '24603.png', '22913.png', '14316.png', '10942.png', '21090.png', '24269.png', '13746.png', '03666.png', '20966.png', '18351.png', '16837.png', '08803.png', '19417.png', '15948.png', '22294.png', '12725.png', '21031.png', '00801.png', '17260.png', '11896.png', '09013.png', '20371.png', '14310.png', '03165.png', '10726.png', '07562.png', '15423.png', '11194.png', '02077.png', '02789.png', '11397.png', '08277.png', '09757.png', '14871.png', '11288.png', '13040.png', '21341.png', '05834.png', '05372.png', '14254.png', '11546.png', '11876.png', '09552.png', '10438.png', '21622.png', '07638.png', '14183.png', '11940.png', '23296.png', '05128.png', '12457.png', '20841.png', '21133.png', '07340.png', '17505.png', '04631.png', '07889.png', '05291.png', '23065.png', '21343.png', '12562.png', '12191.png', '11031.png', '20119.png', '11892.png', '17034.png', '24963.png', '02580.png', '14984.png', '01647.png', '24047.png', '22836.png', '07745.png', '19487.png', '10428.png', '21254.png', '06602.png', '08954.png', '16771.png', '23115.png', '04337.png', '00133.png', '19661.png', '07225.png', '12911.png', '07273.png', '16880.png', '15943.png', '01367.png', '24755.png', '23402.png', '24637.png', '03025.png', '03244.png', '15844.png', '05823.png', '00441.png', '06326.png', '24812.png', '16465.png', '23267.png', '22474.png', '06501.png', '01059.png', '14759.png', '03439.png', '11399.png', '19914.png', '02571.png', '06984.png', '14650.png', '22869.png', '23672.png', '07149.png', '01376.png', '23364.png', '18531.png', '00674.png', '20748.png', '19960.png', '01396.png', '04528.png', '01779.png', '11650.png', '21330.png', '18912.png', '20720.png', '15310.png', '13014.png', '13429.png', '19942.png', '10603.png', '08497.png', '17532.png', '24678.png', '13019.png', '02733.png', '05258.png', '14994.png', '10935.png', '09481.png', '07980.png', '24031.png', '12291.png', '05472.png', '05127.png', '23654.png', '11773.png', '04082.png', '20157.png', '06527.png', '03636.png', '09587.png', '11115.png', '08892.png', '18683.png', '00175.png', '24611.png', '18804.png', '24949.png', '22732.png', '10155.png', '00450.png', '02797.png', '17627.png', '18745.png', '17765.png', '01182.png', '00618.png', '20569.png', '02122.png', '22376.png', '10490.png', '12750.png', '12428.png', '22616.png', '14658.png', '19193.png', '20391.png', '10997.png', '21693.png', '01141.png', '04520.png', '20396.png', '03253.png', '12942.png', '22337.png', '23249.png', '23460.png', '09953.png', '04078.png', '17559.png', '18097.png', '21097.png', '06262.png', '16146.png', '07845.png', '18611.png', '04654.png', '01971.png', '20244.png', '15127.png', '08699.png', '07280.png', '23718.png', '14959.png', '13580.png', '03421.png', '20910.png', '11065.png', '20559.png', '19343.png', '00209.png', '23163.png', '02090.png', '06958.png', '09310.png', '19590.png', '08162.png', '05330.png', '11721.png', '11673.png', '07364.png', '16145.png', '14918.png', '22080.png', '05448.png', '02630.png', '04054.png', '09251.png', '08242.png', '00785.png', '17030.png', '13262.png', '17333.png', '14149.png', '03215.png', '08064.png', '11293.png', '24086.png', '00651.png', '17769.png', '16587.png', '17197.png', '16504.png', '15195.png', '06938.png', '03427.png', '23035.png', '14530.png', '08072.png', '03456.png', '02904.png', '10345.png', '05766.png', '00409.png', '13554.png', '17105.png', '12076.png', '19433.png', '13002.png', '02489.png', '02518.png', '11284.png', '13058.png', '05364.png', '05072.png', '05159.png', '03821.png', '05453.png', '00942.png', '09206.png', '16444.png', '05531.png', '06603.png', '24535.png', '07646.png', '08467.png', '14732.png', '10934.png', '16055.png', '12933.png', '22811.png', '04680.png', '12064.png', '12984.png', '12397.png', '22397.png', '13192.png', '05662.png', '17420.png', '14212.png', '17624.png', '15396.png', '20178.png', '18761.png', '03574.png', '20961.png', '16743.png', '08393.png', '06737.png', '10707.png', '21232.png', '20873.png', '00181.png', '16103.png', '18975.png', '18201.png', '20549.png', '19464.png', '24168.png', '04510.png', '05741.png', '21430.png', '08925.png', '06101.png', '00897.png', '11095.png', '01456.png', '11179.png', '20986.png', '10790.png', '07179.png', '03933.png', '05601.png', '01526.png', '19712.png', '19548.png', '15816.png', '16505.png', '14532.png', '12002.png', '08427.png', '20836.png', '20637.png', '17142.png', '13620.png', '17979.png', '07497.png', '22640.png', '05595.png', '11688.png', '15015.png', '10800.png', '10775.png', '14014.png', '13318.png', '18280.png', '20533.png', '12629.png', '23099.png', '05133.png', '20978.png', '08025.png', '24228.png', '13950.png', '19251.png', '17184.png', '22405.png', '05649.png', '01229.png', '16263.png', '08186.png', '12551.png', '15484.png', '18982.png', '02596.png', '06866.png', '01567.png', '09401.png', '14920.png', '20823.png', '12806.png', '04773.png', '10878.png', '12355.png', '00909.png', '18713.png', '13625.png', '13415.png', '07556.png', '08150.png', '14796.png', '03305.png', '01561.png', '19177.png', '16545.png', '15712.png', '18147.png', '21035.png', '18918.png', '10014.png', '03721.png', '13075.png', '13421.png', '12314.png', '06579.png', '21488.png', '19954.png', '11455.png', '19938.png', '03462.png', '00929.png', '04442.png', '19503.png', '15222.png', '09000.png', '07794.png', '15790.png', '16705.png', '15878.png', '10442.png', '03665.png', '05553.png', '05884.png', '04348.png', '03824.png', '06587.png', '03758.png', '10072.png', '02955.png', '11047.png', '02805.png', '16856.png', '23159.png', '04406.png', '18738.png', '16847.png', '06282.png', '09507.png', '03662.png', '24573.png', '21376.png', '10782.png', '08476.png', '24367.png', '18634.png', '11219.png', '15371.png', '15340.png', '22130.png', '07543.png', '01954.png', '05972.png', '12500.png', '11434.png', '22264.png', '24211.png', '18350.png', '21242.png', '19249.png', '13506.png', '19807.png', '23693.png', '01774.png', '23411.png', '16194.png', '19933.png', '07353.png', '05770.png', '01675.png', '05984.png', '06302.png', '15148.png', '22399.png', '06663.png', '12681.png', '02819.png', '05540.png', '03675.png', '24553.png', '19167.png', '14614.png', '08905.png', '19757.png', '24704.png', '00975.png', '12566.png', '08379.png', '20483.png', '20989.png', '13909.png', '00620.png', '21572.png', '04099.png', '05063.png', '13720.png', '15285.png', '07658.png', '08400.png', '20593.png', '21120.png', '22280.png', '09262.png', '00118.png', '24782.png', '22355.png', '04839.png', '22132.png', '24065.png', '08043.png', '03275.png', '04237.png', '18071.png', '16569.png', '10604.png', '22059.png', '13568.png', '24306.png', '13310.png', '16918.png', '16652.png', '08308.png', '09340.png', '09883.png', '00327.png', '16396.png', '18223.png', '06816.png', '13217.png', '17755.png', '07057.png', '07286.png', '00948.png', '03777.png', '00910.png', '04903.png', '06229.png', '14979.png', '16899.png', '18408.png', '10635.png', '09925.png', '12210.png', '17937.png', '22903.png', '05645.png', '14482.png', '11576.png', '07799.png', '20456.png', '19795.png', '17133.png', '07025.png', '20097.png', '04573.png', '03392.png', '04562.png', '14663.png', '03656.png', '17626.png', '11863.png', '14444.png', '17872.png', '01758.png', '10358.png', '00041.png', '10455.png', '24910.png', '07744.png', '00045.png', '18935.png', '23756.png', '07130.png', '01209.png', '19443.png', '21820.png', '01470.png', '17952.png', '24458.png', '16155.png', '17147.png', '11131.png', '00833.png', '03655.png', '19762.png', '05769.png', '15758.png', '23943.png', '08993.png', '23651.png', '03942.png', '10263.png', '05825.png', '04135.png', '23348.png', '00527.png', '24551.png', '04485.png', '11372.png', '03065.png', '07016.png', '09763.png', '08471.png', '01015.png', '09764.png', '12468.png', '18389.png', '19367.png', '02539.png', '14873.png', '03604.png', '23390.png', '03793.png', '04948.png', '08458.png', '10371.png', '05058.png', '06920.png', '09870.png', '16428.png', '02466.png', '21767.png', '03630.png', '15769.png', '19861.png', '09022.png', '10329.png', '13418.png', '13082.png', '11517.png', '06555.png', '22131.png', '24587.png', '24564.png', '12916.png', '18563.png', '03391.png', '12161.png', '16835.png', '20136.png', '12925.png', '18418.png', '13213.png', '10896.png', '14362.png', '23012.png', '14016.png', '06930.png', '05948.png', '20484.png', '10666.png', '13888.png', '23636.png', '00437.png', '01060.png', '03531.png', '20820.png', '04368.png', '10543.png', '00232.png', '17029.png', '16885.png', '17271.png', '15742.png', '14826.png', '07922.png', '07943.png', '16404.png', '00264.png', '24707.png', '07069.png', '13966.png', '17892.png', '12862.png', '13308.png', '04230.png', '09906.png', '18726.png', '21615.png', '23486.png', '06223.png', '14435.png', '22838.png', '16130.png', '16040.png', '15133.png', '04169.png', '17475.png', '02000.png', '05897.png', '17304.png', '20688.png', '22270.png', '05656.png', '07103.png', '07026.png', '20238.png', '03375.png', '24537.png', '07049.png', '13073.png', '01566.png', '16988.png', '21212.png', '00677.png', '09483.png', '14020.png', '15585.png', '23034.png', '02033.png', '00064.png', '21793.png', '08523.png', '03399.png', '21710.png', '20809.png', '17178.png', '11564.png', '00184.png', '10546.png', '06840.png', '02302.png', '18189.png', '24217.png', '00300.png', '10924.png', '18716.png', '13471.png', '05893.png', '14112.png', '14585.png', '08763.png', '10309.png', '24695.png', '00654.png', '12807.png', '06636.png', '03149.png', '23475.png', '12099.png', '19639.png', '12288.png', '02356.png', '03691.png', '13425.png', '09755.png', '14128.png', '09563.png', '20892.png', '09557.png', '01910.png', '19043.png', '11079.png', '15886.png', '01050.png', '22658.png', '13574.png', '08932.png', '19521.png', '02134.png', '17118.png', '00662.png', '19833.png', '18728.png', '05107.png', '21382.png', '07747.png', '01049.png', '11602.png', '16061.png', '12129.png', '22518.png', '22182.png', '16093.png', '04811.png', '21645.png', '15295.png', '16157.png', '08377.png', '03128.png', '16067.png', '02032.png', '11758.png', '15209.png', '19025.png', '19799.png', '10939.png', '17678.png', '22721.png', '19103.png', '19709.png', '10839.png', '16695.png', '02116.png', '11920.png', '18705.png', '23409.png', '03937.png', '04047.png', '11540.png', '00177.png', '21679.png', '05106.png', '11206.png', '13959.png', '24017.png', '23854.png', '23016.png', '23355.png', '01221.png', '05059.png', '09565.png', '00390.png', '19434.png', '08095.png', '20368.png', '08070.png', '18746.png', '13144.png', '18100.png', '08369.png', '13897.png', '19549.png', '17297.png', '20485.png', '08559.png', '21000.png', '01820.png', '05814.png', '14710.png', '14477.png', '02977.png', '15055.png', '15637.png', '17196.png', '05043.png', '05481.png', '01311.png', '09313.png', '02083.png', '05277.png', '13903.png', '22622.png', '06566.png', '00756.png', '22606.png', '04733.png', '07972.png', '10955.png', '17702.png', '00115.png', '06823.png', '01021.png', '16324.png', '11582.png', '06046.png', '04049.png', '10168.png', '17849.png', '00498.png', '01420.png', '07361.png', '10413.png', '06498.png', '08811.png', '24316.png', '16002.png', '18606.png', '21276.png', '09966.png', '07616.png', '19509.png', '17994.png', '10008.png', '21363.png', '12799.png', '22068.png', '21264.png', '21173.png', '13331.png', '13107.png', '21062.png', '16830.png', '08889.png', '09356.png', '06039.png', '06650.png', '06888.png', '00290.png', '24759.png', '03070.png', '18475.png', '08605.png', '08056.png', '02359.png', '20943.png', '05583.png', '15066.png', '13242.png', '20921.png', '17821.png', '18426.png', '11028.png', '16602.png', '09840.png', '16000.png', '01488.png', '18781.png', '16382.png', '21692.png', '10771.png', '08172.png', '03433.png', '23292.png', '23856.png', '05459.png', '18517.png', '06668.png', '04933.png', '02530.png', '01759.png', '08670.png', '03282.png', '02689.png', '07647.png', '17803.png', '19814.png', '16192.png', '10075.png', '19299.png', '20096.png', '10877.png', '03582.png', '16556.png', '15601.png', '12198.png', '17869.png', '09462.png', '13601.png', '04009.png', '13227.png', '10445.png', '03074.png', '06120.png', '11939.png', '14141.png', '20212.png', '21940.png', '08068.png', '21844.png', '12571.png', '06463.png', '24849.png', '19806.png', '08319.png', '14330.png', '20853.png', '06447.png', '22595.png', '06342.png', '04370.png', '19841.png', '10904.png', '23114.png', '16439.png', '18848.png', '17553.png', '15114.png', '04802.png', '01523.png', '15733.png', '24569.png', '03707.png', '01407.png', '19976.png', '22049.png', '18102.png', '13250.png', '22054.png', '12917.png', '14967.png', '05952.png', '19296.png', '04989.png', '22766.png', '23017.png', '17091.png', '07452.png', '24164.png', '02447.png', '23968.png', '15504.png', '03479.png', '10594.png', '24142.png', '17273.png', '10855.png', '10677.png', '17636.png', '13706.png', '06115.png', '20906.png', '20838.png', '03719.png', '03440.png', '14684.png', '07754.png', '24775.png', '15660.png', '09790.png', '20086.png', '00660.png', '19233.png', '17731.png', '16380.png', '23494.png', '16029.png', '17828.png', '18279.png', '10108.png', '17023.png', '00687.png', '19883.png', '23826.png', '17076.png', '01997.png', '00269.png', '00239.png', '00535.png', '03708.png', '06503.png', '07169.png', '10556.png', '02099.png', '17316.png', '15708.png', '16753.png', '19586.png', '18833.png', '04015.png', '17409.png', '20609.png', '17149.png', '14936.png', '15623.png', '07736.png', '15382.png', '16242.png', '23185.png', '00845.png', '10973.png', '04703.png', '05272.png', '04293.png', '12243.png', '13848.png', '04109.png', '19805.png', '03527.png', '20554.png', '02894.png', '20384.png', '10082.png', '19652.png', '11572.png', '21825.png', '16902.png', '19023.png', '22053.png', '03896.png', '20987.png', '18496.png', '23852.png', '03069.png', '09114.png', '24921.png', '07971.png', '16781.png', '12646.png', '22036.png', '12360.png', '13871.png', '23944.png', '13097.png', '10447.png', '05522.png', '02117.png', '14485.png', '06227.png', '00113.png', '04990.png', '09539.png', '08335.png', '20908.png', '12965.png', '17838.png', '11942.png', '04200.png', '12071.png', '05102.png', '15525.png', '17992.png', '22266.png', '03148.png', '08940.png', '22563.png', '00070.png', '16539.png', '12867.png', '06437.png', '09849.png', '00346.png', '11709.png', '17789.png', '10765.png', '03709.png', '05019.png', '08226.png', '05122.png', '08697.png', '16289.png', '10157.png', '14948.png', '22394.png', '16724.png', '22983.png', '23813.png', '13972.png', '00288.png', '18980.png', '04289.png', '03740.png', '14439.png', '21740.png', '06063.png', '05758.png', '15856.png', '04460.png', '20519.png', '09500.png', '11241.png', '24882.png', '13677.png', '07222.png', '19165.png', '13454.png', '13269.png', '01711.png', '09316.png', '11072.png', '13847.png', '03028.png', '17649.png', '05034.png', '03912.png', '19937.png', '08974.png', '03990.png', '20757.png', '06222.png', '23798.png', '21051.png', '15852.png', '17251.png', '00324.png', '23105.png', '06694.png', '20655.png', '17356.png', '07627.png', '02812.png', '09707.png', '13659.png', '20168.png', '12470.png', '17657.png', '12648.png', '16185.png', '16759.png', '11166.png', '14174.png', '05862.png', '04417.png', '21773.png', '14860.png', '10545.png', '09210.png', '04557.png', '12393.png', '14682.png', '00607.png', '13923.png', '12634.png', '20594.png', '08825.png', '07248.png', '24864.png', '06083.png', '19714.png', '24960.png', '09781.png', '11002.png', '12412.png', '13393.png', '23974.png', '08760.png', '20455.png', '21841.png', '01947.png', '19987.png', '20326.png', '00754.png', '14352.png', '02697.png', '00259.png', '15981.png', '20183.png', '01432.png', '09353.png', '02112.png', '24494.png', '05035.png', '11643.png', '09365.png', '04472.png', '13687.png', '12714.png', '14621.png', '13005.png', '13603.png', '06200.png', '12386.png', '21214.png', '13145.png', '20201.png', '00795.png', '05750.png', '11505.png', '21781.png', '04374.png', '09426.png', '09699.png', '02473.png', '24797.png', '18632.png', '15500.png', '04775.png', '14730.png', '10728.png', '01452.png', '23368.png', '24362.png', '15204.png', '03816.png', '00314.png', '22165.png', '19050.png', '10700.png', '17780.png', '16876.png', '02486.png', '16740.png', '13287.png', '02678.png', '07104.png', '17305.png', '01556.png', '08874.png', '00038.png', '11565.png', '16044.png', '18021.png', '05580.png', '08741.png', '05679.png', '04208.png', '10664.png', '22272.png', '02464.png', '05539.png', '04316.png', '09231.png', '00664.png', '23359.png', '22856.png', '05965.png', '20144.png', '21252.png', '09375.png', '06834.png', '21146.png', '20454.png', '07818.png', '03690.png', '19382.png', '03580.png', '23313.png', '18436.png', '02205.png', '16319.png', '18182.png', '11504.png', '05335.png', '22084.png', '07087.png', '08696.png', '22980.png', '10366.png', '12419.png', '04131.png', '12783.png', '17498.png', '10460.png', '01110.png', '05653.png', '09311.png', '13313.png', '22351.png', '02804.png', '09971.png', '17378.png', '12912.png', '10281.png', '14738.png', '09979.png', '18844.png', '01431.png', '08876.png', '08684.png', '02579.png', '14543.png', '11156.png', '11177.png', '14440.png', '14683.png', '21060.png', '04677.png', '04881.png', '06726.png', '07899.png', '22720.png', '24075.png', '17080.png', '01100.png', '13079.png', '09330.png', '03517.png', '11554.png', '17489.png', '12727.png', '24081.png', '09804.png', '18981.png', '00952.png', '04849.png', '03929.png', '21778.png', '12816.png', '20957.png', '23560.png', '16424.png', '23028.png', '06721.png', '10230.png', '18997.png', '04069.png', '07379.png', '14328.png', '17306.png', '24565.png', '04523.png', '13299.png', '17535.png', '13148.png', '17046.png', '18292.png', '24242.png', '22945.png', '15674.png', '19878.png', '23912.png', '01872.png', '20169.png', '23556.png', '10124.png', '22223.png', '22854.png', '14303.png', '24640.png', '17834.png', '01301.png', '08285.png', '16905.png', '03217.png', '05421.png', '17422.png', '06292.png', '04991.png', '22073.png', '13890.png', '06571.png', '19693.png', '06484.png', '23084.png', '16894.png', '05066.png', '21516.png', '03902.png', '02196.png', '15233.png', '02622.png', '01516.png', '17112.png', '15168.png', '09498.png', '02169.png', '05294.png', '15734.png', '22874.png', '15724.png', '21950.png', '20812.png', '23570.png', '24799.png', '04213.png', '15781.png', '06942.png', '23522.png', '21729.png', '13830.png', '15517.png', '21534.png', '07867.png', '09253.png', '18620.png', '06605.png', '15801.png', '23222.png', '02132.png', '23167.png', '00446.png', '14343.png', '03863.png', '05986.png', '24191.png', '07005.png', '22007.png', '14747.png', '06038.png', '22940.png', '15178.png', '16012.png', '20435.png', '19300.png', '11996.png', '24737.png', '06030.png', '04149.png', '03096.png', '22891.png', '19425.png', '14405.png', '14702.png', '20780.png', '02216.png', '02394.png', '12318.png', '13627.png', '23330.png', '23250.png', '19794.png', '15788.png', '12921.png', '20828.png', '06118.png', '03509.png', '06687.png', '24235.png', '22555.png', '06119.png', '12924.png', '10807.png', '23149.png', '22231.png', '17717.png', '16269.png', '19550.png', '04824.png', '04134.png', '13824.png', '14041.png', '21522.png', '15470.png', '22826.png', '14107.png', '20217.png', '01332.png', '05429.png', '00236.png', '03997.png', '03256.png', '05369.png', '22608.png', '20055.png', '18478.png', '23482.png', '18073.png', '03890.png', '19214.png', '24740.png', '24677.png', '23329.png', '21597.png', '17121.png', '03484.png', '16041.png', '18060.png', '10913.png', '22241.png', '17295.png', '15830.png', '14269.png', '17796.png', '03724.png', '20608.png', '20693.png', '05736.png', '21513.png', '00163.png', '08404.png', '03872.png', '05921.png', '03741.png', '11871.png', '12381.png', '00670.png', '06935.png', '15732.png', '21056.png', '15203.png', '14406.png', '21652.png', '12447.png', '18521.png', '02336.png', '01400.png', '07522.png', '20337.png', '10275.png', '20190.png', '04056.png', '12433.png', '05696.png', '10184.png', '11492.png', '15412.png', '23714.png', '13182.png', '22326.png', '01477.png', '06204.png', '15415.png', '00586.png', '10584.png', '16476.png', '21483.png', '22033.png', '12283.png', '06329.png', '03623.png', '19376.png', '08640.png', '09612.png', '10267.png', '18974.png', '14329.png', '10837.png', '14576.png', '23529.png', '01771.png', '17436.png', '12944.png', '05976.png', '08318.png', '02352.png', '06041.png', '15635.png', '19399.png', '18821.png', '15996.png', '19240.png', '19062.png', '06800.png', '20540.png', '05268.png', '15961.png', '20682.png', '15459.png', '10780.png', '06727.png', '20544.png', '08488.png', '04577.png', '06753.png', '24757.png', '05346.png', '17469.png', '22495.png', '06446.png', '12533.png', '19315.png', '23351.png', '21945.png', '11476.png', '14052.png', '13765.png', '16344.png', '04381.png', '20504.png', '02186.png', '15865.png', '21441.png', '06429.png', '02280.png', '06733.png', '12949.png', '20550.png', '13176.png', '10344.png', '01398.png', '13566.png', '06757.png', '10731.png', '07486.png', '10751.png', '11699.png', '11790.png', '24938.png', '11460.png', '01472.png', '00182.png', '11916.png', '13984.png', '18868.png', '21069.png', '20202.png', '09358.png', '03987.png', '17870.png', '10254.png', '01979.png', '15952.png', '05112.png', '16236.png', '07665.png', '10458.png', '19138.png', '04559.png', '06873.png', '01125.png', '19007.png', '02457.png', '16052.png', '17961.png', '11453.png', '16422.png', '00718.png', '17265.png', '07756.png', '22311.png', '00411.png', '24002.png', '15704.png', '07153.png', '11930.png', '03751.png', '21092.png', '17137.png', '10305.png', '14892.png', '10824.png', '16181.png', '12015.png', '16665.png', '09984.png', '00546.png', '21250.png', '15003.png', '16412.png', '21944.png', '05946.png', '15060.png', '03914.png', '13230.png', '23122.png', '17152.png', '23319.png', '13754.png', '13936.png', '17653.png', '06205.png', '23699.png', '05549.png', '09495.png', '13618.png', '15890.png', '13651.png', '19154.png', '02002.png', '11357.png', '23590.png', '16414.png', '06163.png', '00408.png', '04106.png', '06435.png', '07299.png', '19174.png', '12992.png', '09053.png', '02832.png', '08145.png', '22961.png', '20789.png', '19527.png', '11821.png', '04652.png', '08350.png', '22108.png', '03730.png', '20996.png', '00710.png', '12967.png', '09162.png', '02437.png', '06403.png', '17454.png', '07596.png', '06390.png', '10987.png', '19246.png', '03670.png', '09387.png', '14708.png', '09158.png', '09032.png', '00040.png', '20012.png', '18919.png', '00694.png', '10192.png', '09511.png', '05252.png', '00135.png', '15569.png', '06741.png', '10747.png', '13160.png', '06426.png', '10535.png', '01717.png', '02893.png', '20624.png', '15017.png', '17218.png', '07740.png', '17185.png', '07035.png', '15453.png', '18101.png', '22993.png', '03754.png', '02008.png', '00626.png', '19951.png', '20694.png', '01635.png', '20482.png', '24594.png', '19360.png', '18437.png', '24419.png', '22204.png', '20022.png', '05494.png', '17888.png', '11487.png', '17199.png', '18427.png', '08591.png', '22501.png', '02973.png', '12109.png', '10720.png', '02885.png', '18121.png', '16533.png', '14983.png', '22324.png', '09948.png', '18503.png', '09941.png', '11919.png', '11671.png', '17453.png', '07427.png', '21796.png', '07851.png', '01902.png', '17973.png', '01547.png', '18307.png', '07431.png', '00619.png', '01038.png', '16160.png', '03459.png', '07332.png', '21730.png', '21996.png', '23548.png', '21274.png', '21532.png', '05956.png', '13047.png', '03614.png', '11419.png', '00544.png', '08676.png', '03054.png', '17592.png', '06787.png', '13394.png', '00937.png', '04086.png', '10293.png', '20638.png', '17607.png', '19030.png', '21728.png', '07895.png', '11478.png', '23485.png', '22406.png', '23503.png', '18049.png', '14763.png', '13254.png', '10949.png', '01687.png', '16565.png', '07654.png', '21984.png', '09819.png', '18734.png', '10961.png', '24274.png', '20863.png', '02780.png', '15141.png', '09926.png', '13441.png', '06211.png', '21378.png', '00590.png', '15053.png', '10902.png', '18033.png', '17072.png', '20449.png', '21754.png', '10332.png', '04007.png', '09668.png', '20499.png', '03417.png', '01152.png', '04290.png', '21837.png', '09303.png', '10277.png', '16291.png', '17590.png', '13908.png', '06841.png', '07538.png', '00415.png', '12027.png', '12886.png', '14370.png', '05403.png', '11661.png', '20063.png', '00344.png', '08445.png', '11943.png', '19274.png', '13445.png', '24176.png', '07146.png', '09437.png', '24669.png', '02693.png', '07368.png', '19194.png', '13317.png', '19978.png', '21036.png', '00935.png', '13434.png', '11927.png', '21124.png', '13186.png', '18854.png', '11380.png', '08733.png', '00037.png', '21964.png', '05599.png', '23042.png', '09760.png', '07308.png', '06134.png', '23265.png', '00417.png', '11988.png', '22605.png', '24356.png', '20945.png', '15095.png', '15057.png', '20990.png', '01201.png', '14785.png', '14098.png', '24504.png', '17466.png', '03581.png', '08011.png', '23940.png', '03976.png', '02218.png', '02409.png', '05477.png', '07893.png', '16887.png', '15568.png', '05772.png', '12752.png', '09685.png', '09997.png', '18240.png', '23063.png', '15458.png', '04670.png', '01620.png', '24013.png', '12449.png', '21230.png', '18495.png', '23062.png', '00307.png', '17503.png', '10998.png', '06950.png', '13000.png', '00715.png', '24344.png', '02516.png', '15777.png', '02610.png', '08950.png', '00480.png', '03299.png', '08122.png', '01596.png', '15700.png', '11017.png', '01106.png', '01081.png', '09523.png', '05856.png', '08791.png', '03464.png', '06546.png', '09545.png', '03732.png', '14603.png', '02163.png', '13039.png', '23362.png', '07951.png', '05659.png', '04558.png', '12583.png', '23838.png', '11846.png', '06054.png', '16483.png', '01070.png', '18692.png', '05382.png', '00956.png', '17942.png', '05390.png', '01364.png', '19019.png', '01468.png', '24930.png', '01313.png', '19363.png', '10427.png', '18727.png', '19145.png', '01836.png', '22946.png', '16209.png', '16506.png', '02184.png', '05171.png', '16437.png', '10246.png', '23763.png', '21772.png', '08090.png', '19354.png', '07355.png', '13538.png', '17007.png', '08170.png', '14592.png', '04581.png', '01076.png', '19803.png', '24171.png', '00773.png', '10018.png', '12098.png', '18267.png', '00787.png', '20150.png', '00971.png', '21328.png', '18846.png', '07661.png', '07735.png', '13921.png', '01146.png', '23676.png', '04646.png', '23238.png', '05928.png', '01433.png', '21618.png', '00158.png', '09591.png', '11739.png', '11472.png', '04730.png', '22842.png', '17347.png', '03113.png', '15574.png', '14777.png', '18453.png', '01768.png', '04956.png', '14635.png', '20913.png', '13205.png', '24795.png', '16645.png', '22331.png', '09808.png', '18479.png', '03996.png', '23928.png', '02688.png', '14347.png', '08562.png', '18879.png', '02154.png', '22208.png', '09006.png', '01856.png', '13172.png', '11972.png', '14655.png', '09224.png', '24147.png', '01169.png', '12811.png', '22034.png', '24831.png', '05808.png', '01684.png', '16966.png', '23398.png', '03485.png', '21918.png', '23864.png', '07872.png', '02079.png', '02760.png', '10410.png', '00579.png', '03705.png', '11272.png', '21958.png', '00143.png', '07118.png', '20742.png']
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

        if i_iter == 282529:
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
