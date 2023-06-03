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
    a = ['21536.png', '11728.png', '18856.png', '00491.png', '07293.png', '06964.png', '20877.png', '14849.png', '11301.png', '14914.png', '18530.png', '02232.png', '01709.png', '03729.png', '06087.png', '10917.png', '15050.png', '13717.png', '06392.png', '02652.png', '07538.png', '13974.png', '21410.png', '23832.png', '15747.png', '06261.png', '03480.png', '13321.png', '14886.png', '23916.png', '09160.png', '02891.png', '08425.png', '10454.png', '14434.png', '03636.png', '23937.png', '02653.png', '00828.png', '21733.png', '23014.png', '15298.png', '14916.png', '23291.png', '17108.png', '21521.png', '22909.png', '08671.png', '03380.png', '08140.png', '13412.png', '18693.png', '05700.png', '03174.png', '22990.png', '24138.png', '09158.png', '04033.png', '20981.png', '12855.png', '18681.png', '07876.png', '20588.png', '10787.png', '14206.png', '23332.png', '21528.png', '04130.png', '09766.png', '16153.png', '00527.png', '06627.png', '14895.png', '07923.png', '18123.png', '18854.png', '08763.png', '04185.png', '07209.png', '18267.png', '18014.png', '07530.png', '17857.png', '03693.png', '09448.png', '23990.png', '04220.png', '13655.png', '15159.png', '24284.png', '08351.png', '10466.png', '01819.png', '11564.png', '16837.png', '24117.png', '20501.png', '13476.png', '23301.png', '02732.png', '23068.png', '05423.png', '07199.png', '19458.png', '06866.png', '01192.png', '23524.png', '20449.png', '05107.png', '24705.png', '01458.png', '08185.png', '04362.png', '05547.png', '07315.png', '13009.png', '04586.png', '19642.png', '02575.png', '07247.png', '06641.png', '01461.png', '03087.png', '17389.png', '07173.png', '10718.png', '19173.png', '17547.png', '05213.png', '19466.png', '09031.png', '16380.png', '08641.png', '09434.png', '03903.png', '11914.png', '14120.png', '06827.png', '13554.png', '17819.png', '09846.png', '15142.png', '22875.png', '18960.png', '06003.png', '01354.png', '23744.png', '04349.png', '07351.png', '21967.png', '08789.png', '13929.png', '01923.png', '03603.png', '15706.png', '09588.png', '08355.png', '07374.png', '02937.png', '09636.png', '15030.png', '24174.png', '24489.png', '04249.png', '13959.png', '20719.png', '03054.png', '24827.png', '06136.png', '23687.png', '14727.png', '00399.png', '01049.png', '18965.png', '17914.png', '08839.png', '03496.png', '12306.png', '18223.png', '08339.png', '22842.png', '21780.png', '15580.png', '04681.png', '09162.png', '19855.png', '12110.png', '04176.png', '13353.png', '19162.png', '13259.png', '04799.png', '17802.png', '24349.png', '12874.png', '17721.png', '07763.png', '19679.png', '12406.png', '19812.png', '15854.png', '18240.png', '09672.png', '16437.png', '11689.png', '01052.png', '01677.png', '09603.png', '22734.png', '12452.png', '14111.png', '24163.png', '18428.png', '05573.png', '23310.png', '01917.png', '07811.png', '11509.png', '19005.png', '15206.png', '12008.png', '22669.png', '05182.png', '08965.png', '11915.png', '20440.png', '06669.png', '24149.png', '01074.png', '15216.png', '00962.png', '14329.png', '11132.png', '23756.png', '11706.png', '03161.png', '13182.png', '17086.png', '14043.png', '06027.png', '11784.png', '22633.png', '05602.png', '01343.png', '04348.png', '12860.png', '11514.png', '01466.png', '16255.png', '11922.png', '06374.png', '02308.png', '16427.png', '08406.png', '01002.png', '00925.png', '14688.png', '01718.png', '02333.png', '10824.png', '01066.png', '20687.png', '05221.png', '14036.png', '17871.png', '11608.png', '06497.png', '00298.png', '23138.png', '04055.png', '11668.png', '01654.png', '03812.png', '11128.png', '04514.png', '23356.png', '14779.png', '23045.png', '20887.png', '03516.png', '12370.png', '09333.png', '01327.png', '15959.png', '07120.png', '20220.png', '24205.png', '11830.png', '19110.png', '06161.png', '10376.png', '03571.png', '08093.png', '08787.png', '21595.png', '23026.png', '00363.png', '11395.png', '05510.png', '07128.png', '24493.png', '17473.png', '04312.png', '12190.png', '19429.png', '19016.png', '12543.png', '15446.png', '10315.png', '06870.png', '08726.png', '22435.png', '03503.png', '01804.png', '19308.png', '06946.png', '22483.png', '09983.png', '00039.png', '17162.png', '21662.png', '12734.png', '02031.png', '16478.png', '02991.png', '07371.png', '15140.png', '20944.png', '02256.png', '09880.png', '04222.png', '23977.png', '20051.png', '20174.png', '20167.png', '08029.png', '04530.png', '18076.png', '09536.png', '06111.png', '21429.png', '12961.png', '20001.png', '12653.png', '15944.png', '06676.png', '16825.png', '16739.png', '08797.png', '03616.png', '11050.png', '20461.png', '19724.png', '09314.png', '17234.png', '02222.png', '09105.png', '19871.png', '22446.png', '14578.png', '17511.png', '14749.png', '01625.png', '18807.png', '01359.png', '14680.png', '17484.png', '18700.png', '02705.png', '00940.png', '22275.png', '15975.png', '23000.png', '00109.png', '13851.png', '11061.png', '03110.png', '01495.png', '18136.png', '11576.png', '06438.png', '06908.png', '23089.png', '06632.png', '01006.png', '10978.png', '17588.png', '11071.png', '14671.png', '02241.png', '24172.png', '01986.png', '13675.png', '24752.png', '07761.png', '18356.png', '15896.png', '07328.png', '22809.png', '18752.png', '02929.png', '05396.png', '17084.png', '14279.png', '00529.png', '17378.png', '11839.png', '24873.png', '13766.png', '11086.png', '08341.png', '15067.png', '22814.png', '11027.png', '17708.png', '24557.png', '13007.png', '01703.png', '24516.png', '00274.png', '15723.png', '22792.png', '02599.png', '11651.png', '23952.png', '13028.png', '03018.png', '14687.png', '00586.png', '17302.png', '23260.png', '03878.png', '12895.png', '10391.png', '07355.png', '13867.png', '03250.png', '17922.png', '23215.png', '00105.png', '20631.png', '04882.png', '11686.png', '04535.png', '08147.png', '02345.png', '20572.png', '07741.png', '04386.png', '06620.png', '04860.png', '00860.png', '07872.png', '14514.png', '11485.png', '05112.png', '01122.png', '04525.png', '01918.png', '13364.png', '21988.png', '22162.png', '20280.png', '02403.png', '09481.png', '10385.png', '23022.png', '11318.png', '04388.png', '06675.png', '08933.png', '10122.png', '08342.png', '18512.png', '09921.png', '12939.png', '19683.png', '12284.png', '21301.png', '08020.png', '02288.png', '05212.png', '24189.png', '15281.png', '06153.png', '02810.png', '07286.png', '10089.png', '14455.png', '17865.png', '23376.png', '10271.png', '04370.png', '03819.png', '21040.png', '10669.png', '21404.png', '00997.png', '17277.png', '04675.png', '03944.png', '05817.png', '17906.png', '21332.png', '11221.png', '06778.png', '20525.png', '20046.png', '17561.png', '20859.png', '09958.png', '24772.png', '10063.png', '08692.png', '15578.png', '09378.png', '10284.png', '04491.png', '09308.png', '08655.png', '16986.png', '17101.png', '04268.png', '24002.png', '12581.png', '03760.png', '21825.png', '16849.png', '18642.png', '09936.png', '12955.png', '11755.png', '09280.png', '23101.png', '20787.png', '12348.png', '18909.png', '21129.png', '19934.png', '18968.png', '04282.png', '08283.png', '10939.png', '19522.png', '10627.png', '03225.png', '22471.png', '22371.png', '01601.png', '02062.png', '21934.png', '06464.png', '21038.png', '16440.png', '13037.png', '15766.png', '07708.png', '02541.png', '20082.png', '16305.png', '22462.png', '02249.png', '03647.png', '15618.png', '01971.png', '19841.png', '17310.png', '02714.png', '08378.png', '03140.png', '14568.png', '05751.png', '19205.png', '23004.png', '13490.png', '20769.png', '18553.png', '07235.png', '00222.png', '02981.png', '05858.png', '16463.png', '14717.png', '00125.png', '11162.png', '23706.png', '11331.png', '10027.png', '19610.png', '20098.png', '20865.png', '12960.png', '05299.png', '13886.png', '00707.png', '24450.png', '06050.png', '09206.png', '02750.png', '03977.png', '15259.png', '15000.png', '09163.png', '17440.png', '24715.png', '04887.png', '01537.png', '24676.png', '00313.png', '23596.png', '16488.png', '22203.png', '24535.png', '06886.png', '18918.png', '08792.png', '11033.png', '24906.png', '05546.png', '15061.png', '13751.png', '04870.png', '02311.png', '05460.png', '05634.png', '04208.png', '24695.png', '02094.png', '05415.png', '01952.png', '18293.png', '02425.png', '03510.png', '04405.png', '21587.png', '07730.png', '08232.png', '11228.png', '17014.png', '08045.png', '15350.png', '24740.png', '02789.png', '02530.png', '22519.png', '07226.png', '00772.png', '01833.png', '22558.png', '06923.png', '13673.png', '08245.png', '10531.png', '24444.png', '12136.png', '00406.png', '08129.png', '11678.png', '16040.png', '20283.png', '00231.png', '18670.png', '20303.png', '18365.png', '23361.png', '02801.png', '18005.png', '09969.png', '08172.png', '05003.png', '04161.png', '23828.png', '09530.png', '10144.png', '03677.png', '23953.png', '19178.png', '19380.png', '22577.png', '08666.png', '16933.png', '14473.png', '10608.png', '24471.png', '21284.png', '11198.png', '12612.png', '12999.png', '05823.png', '00888.png', '11555.png', '08030.png', '16234.png', '18039.png', '17300.png', '09769.png', '11835.png', '18081.png', '01697.png', '18337.png', '17338.png', '12214.png', '02199.png', '07667.png', '21991.png', '16860.png', '04660.png', '23193.png', '03907.png', '22400.png', '15692.png', '18003.png', '17373.png', '00333.png', '05440.png', '00735.png', '18082.png', '11200.png', '07063.png', '18023.png', '11445.png', '01502.png', '18148.png', '04089.png', '04035.png', '06015.png', '08875.png', '17265.png', '08611.png', '18825.png', '05021.png', '17003.png', '03190.png', '13931.png', '17119.png', '02016.png', '05016.png', '19337.png', '19740.png', '14383.png', '20160.png', '07216.png', '14124.png', '17307.png', '11738.png', '12373.png', '11041.png', '00784.png', '15368.png', '00743.png', '18962.png', '17194.png', '07630.png', '16862.png', '23024.png', '18046.png', '20678.png', '11579.png', '06680.png', '18931.png', '11405.png', '01396.png', '00915.png', '01539.png', '01065.png', '05425.png', '23636.png', '19464.png', '18417.png', '21183.png', '00027.png', '23130.png', '22715.png', '16472.png', '20421.png', '12299.png', '12883.png', '05566.png', '07568.png', '08502.png', '03354.png', '21987.png', '23979.png', '13274.png', '07662.png', '16921.png', '00160.png', '06297.png', '02246.png', '16373.png', '16575.png', '05337.png', '24899.png', '20758.png', '13271.png', '09553.png', '01767.png', '14105.png', '11334.png', '00575.png', '04180.png', '00324.png', '24595.png', '15084.png', '11665.png', '14610.png', '24891.png', '02547.png', '19081.png', '07394.png', '17291.png', '23148.png', '24659.png', '15509.png', '14839.png', '14053.png', '20895.png', '18984.png', '18517.png', '13477.png', '17610.png', '24931.png', '21866.png', '05907.png', '01453.png', '21795.png', '05641.png', '24682.png', '00916.png', '04204.png', '24558.png', '02611.png', '06690.png', '15307.png', '04822.png', '21413.png', '18551.png', '15592.png', '07276.png', '07850.png', '13324.png', '09291.png', '13654.png', '00809.png', '21682.png', '10051.png', '20373.png', '16774.png', '22374.png', '07002.png', '11874.png', '03705.png', '09490.png', '17548.png', '11207.png', '05652.png', '07637.png', '24691.png', '07468.png', '06823.png', '03894.png', '05738.png', '11888.png', '23423.png', '10230.png', '13483.png', '21739.png', '23901.png', '24645.png', '19717.png', '23435.png', '05873.png', '01881.png', '14025.png', '16293.png', '09928.png', '21567.png', '12317.png', '06270.png', '11699.png', '01351.png', '23919.png', '03798.png', '19523.png', '13417.png', '06674.png', '16315.png', '10786.png', '04192.png', '06069.png', '11660.png', '18676.png', '10425.png', '06844.png', '02733.png', '06557.png', '16159.png', '05067.png', '13927.png', '06276.png', '10204.png', '03642.png', '15532.png', '18701.png', '15659.png', '00905.png', '04301.png', '11866.png', '06569.png', '01180.png', '22340.png', '05342.png', '01226.png', '00869.png', '20121.png', '06651.png', '17742.png', '19715.png', '14823.png', '07854.png', '15810.png', '07213.png', '04183.png', '22737.png', '08539.png', '22678.png', '15697.png', '13458.png', '20628.png', '24384.png', '15673.png', '13883.png', '20849.png', '03986.png', '10294.png', '04552.png', '13606.png', '24462.png', '04661.png', '22873.png', '03793.png', '20848.png', '14076.png', '12283.png', '17186.png', '24158.png', '24215.png', '24510.png', '20689.png', '22834.png', '16632.png', '24423.png', '05426.png', '12140.png', '03833.png', '09155.png', '18635.png', '06764.png', '20564.png', '05125.png', '08518.png', '02676.png', '06461.png', '11364.png', '07440.png', '01398.png', '08470.png', '02072.png', '09660.png', '18089.png', '13812.png', '00084.png', '24862.png', '02464.png', '22647.png', '16802.png', '19352.png', '03607.png', '14562.png', '14475.png', '17704.png', '10560.png', '16336.png', '12470.png', '13999.png', '08057.png', '20546.png', '19228.png', '14169.png', '03375.png', '03939.png', '03108.png', '22593.png', '20933.png', '20668.png', '23536.png', '03384.png', '19925.png', '24074.png', '21237.png', '24915.png', '18027.png', '18418.png', '04733.png', '21031.png', '00827.png', '12733.png', '12384.png', '07821.png', '15782.png', '22005.png', '14209.png', '23591.png', '05450.png', '19619.png', '18487.png', '23223.png', '16708.png', '12681.png', '02776.png', '07147.png', '23948.png', '17758.png', '07049.png', '15760.png', '06879.png', '09245.png', '21285.png', '00099.png', '07603.png', '17658.png', '15510.png', '08821.png', '05956.png', '14591.png', '02063.png', '02109.png', '19314.png', '24761.png', '09753.png', '04588.png', '19336.png', '12326.png', '06767.png', '02493.png', '00779.png', '07112.png', '01723.png', '13825.png', '10634.png', '21270.png', '13374.png', '24274.png', '07840.png', '05715.png', '17554.png', '14558.png', '16275.png', '01094.png', '20800.png', '05190.png', '05084.png', '13617.png', '05779.png', '12837.png', '07597.png', '22527.png', '10178.png', '24393.png', '17475.png', '22637.png', '22605.png', '18144.png', '01764.png', '20879.png', '22409.png', '11997.png', '03917.png', '10730.png', '21654.png', '15888.png', '13646.png', '08521.png', '06149.png', '05106.png', '19649.png', '05442.png', '14234.png', '19411.png', '02914.png', '09083.png', '24671.png', '13236.png', '13913.png', '21758.png', '02831.png', '17016.png', '11203.png', '05515.png', '20487.png', '17988.png', '13241.png', '00868.png', '02769.png', '24892.png', '17817.png', '19220.png', '17664.png', '08901.png', '15560.png', '03634.png', '22751.png', '14745.png', '17569.png', '02601.png', '17146.png', '10952.png', '10017.png', '16374.png', '20268.png', '18957.png', '23329.png', '01201.png', '16977.png', '10766.png', '10365.png', '17181.png', '14029.png', '19149.png', '23264.png', '17429.png', '15234.png', '04065.png', '00034.png', '13198.png', '16037.png', '15374.png', '23998.png', '05344.png', '21657.png', '02573.png', '03499.png', '15897.png', '07501.png', '22266.png', '22109.png', '08920.png', '17533.png', '14231.png', '24240.png', '16758.png', '12873.png', '08858.png', '23437.png', '17224.png', '04747.png', '22800.png', '12189.png', '15753.png', '18084.png', '06654.png', '15708.png', '05316.png', '11412.png', '06421.png', '09310.png', '05078.png', '15312.png', '07297.png', '16704.png', '15535.png', '23384.png', '13824.png', '22080.png', '15058.png', '05593.png', '02007.png', '05853.png', '22141.png', '22723.png', '22321.png', '09664.png', '05824.png', '10128.png', '08600.png', '06305.png', '11341.png', '08110.png', '20722.png', '23749.png', '01791.png', '18744.png', '16151.png', '08117.png', '02845.png', '08886.png', '05354.png', '02212.png', '12112.png', '21033.png', '09584.png', '18763.png', '23122.png', '15917.png', '24219.png', '04693.png', '12588.png', '16326.png', '11351.png', '04170.png', '08831.png', '07805.png', '03710.png', '18088.png', '07292.png', '21822.png', '15595.png', '07074.png', '12017.png', '09577.png', '19034.png', '14413.png', '02999.png', '03377.png', '08900.png', '02882.png', '07948.png', '04261.png', '12819.png', '12129.png', '00493.png', '07856.png', '09225.png', '24932.png', '23920.png', '04464.png', '16997.png', '07734.png', '13714.png', '06933.png', '00704.png', '00097.png', '18731.png', '02763.png', '08216.png', '09464.png', '20432.png', '11447.png', '23188.png', '04580.png', '11851.png', '12931.png', '08837.png', '23136.png', '02577.png', '15023.png', '03835.png', '23164.png', '06083.png', '24408.png', '04381.png', '05469.png', '00783.png', '00531.png', '23117.png', '14672.png', '24182.png', '16642.png', '05503.png', '07729.png', '22924.png', '15587.png', '23333.png', '15531.png', '05830.png', '12952.png', '02640.png', '16252.png', '17078.png', '24687.png', '07450.png', '24554.png', '16461.png', '08330.png', '08947.png', '09106.png', '01584.png', '09993.png', '06300.png', '10048.png', '00044.png', '09712.png', '12785.png', '19292.png', '04090.png', '05987.png', '18340.png', '03485.png', '22826.png', '13578.png', '07868.png', '04437.png', '16007.png', '20975.png', '00150.png', '22159.png', '11829.png', '24882.png', '16312.png', '11695.png', '11239.png', '06415.png', '02795.png', '16697.png', '08987.png', '12654.png', '13159.png', '10007.png', '09979.png', '12445.png', '03246.png', '20620.png', '05304.png', '16922.png', '24464.png', '18562.png', '17619.png', '05684.png', '01017.png', '03943.png', '00426.png', '02989.png', '23874.png', '04512.png', '20277.png', '04156.png', '16351.png', '12196.png', '21143.png', '10606.png', '20133.png', '20331.png', '09404.png', '03770.png', '19593.png', '21267.png', '13565.png', '09614.png', '20275.png', '00843.png', '21341.png', '06299.png', '08192.png', '02497.png', '19453.png', '09407.png', '12049.png', '08334.png', '06306.png', '13875.png', '15163.png', '23943.png', '05482.png', '00470.png', '03627.png', '09171.png', '03991.png', '23418.png', '17362.png', '15035.png', '20704.png', '20221.png', '14982.png', '13590.png', '00760.png', '20087.png', '01476.png', '04281.png', '08306.png', '23941.png', '20700.png', '03857.png', '21119.png', '04740.png', '18262.png', '12509.png', '17178.png', '16647.png', '21512.png', '13950.png', '22507.png', '13823.png', '05626.png', '08304.png', '08907.png', '08564.png', '09309.png', '02796.png', '16805.png', '15661.png', '01778.png', '04213.png', '16800.png', '02853.png', '14002.png', '08593.png', '10809.png', '19652.png', '06993.png', '24726.png', '17180.png', '24661.png', '14825.png', '07398.png', '20932.png', '01080.png', '18923.png', '19505.png', '22332.png', '18770.png', '06463.png', '13840.png', '05762.png', '03752.png', '01225.png', '20585.png', '10134.png', '08613.png', '06925.png', '17337.png', '05235.png', '06822.png', '14618.png', '07598.png', '21980.png', '12219.png', '14004.png', '04526.png', '12257.png', '21794.png', '21927.png', '10595.png', '00561.png', '17941.png', '02420.png', '24181.png', '14669.png', '12381.png', '02950.png', '13552.png', '16214.png', '09463.png', '01946.png', '04519.png', '16896.png', '22817.png', '12918.png', '16240.png', '03121.png', '21419.png', '23684.png', '24173.png', '02792.png', '02350.png', '23912.png', '02971.png', '19093.png', '23030.png', '17725.png', '04621.png', '05034.png', '24903.png', '20097.png', '21234.png', '21979.png', '20169.png', '10849.png', '19685.png', '04337.png', '07666.png', '02485.png', '18271.png', '01480.png', '07340.png', '04110.png', '08091.png', '20351.png', '07185.png', '04673.png', '23398.png', '02490.png', '15809.png', '24000.png', '01153.png', '07552.png', '11966.png', '08075.png', '18139.png', '14022.png', '12157.png', '23336.png', '13253.png', '12223.png', '08139.png', '21878.png', '04206.png', '16614.png', '06997.png', '24685.png', '21416.png', '22503.png', '08650.png', '24190.png', '03206.png', '19629.png', '14520.png', '20179.png', '00459.png', '11017.png', '03181.png', '21153.png', '24011.png', '02457.png', '21217.png', '21168.png', '15504.png', '15346.png', '10334.png', '21403.png', '17327.png', '03831.png', '19318.png', '08369.png', '05535.png', '17852.png', '13487.png', '08153.png', '14457.png', '15141.png', '18447.png', '18461.png', '23525.png', '14755.png', '14110.png', '00299.png', '01156.png', '16180.png', '00082.png', '05861.png', '15267.png', '05688.png', '20347.png', '06843.png', '16274.png', '21302.png', '21346.png', '21940.png', '19216.png', '13070.png', '00866.png', '18220.png', '05400.png', '20746.png', '19214.png', '02038.png', '17727.png', '04417.png', '00122.png', '12753.png', '10528.png', '16752.png', '24517.png', '13054.png', '04230.png', '10616.png', '06926.png', '02454.png', '03052.png', '12052.png', '22426.png', '07623.png', '10997.png', '04687.png', '02211.png', '22550.png', '20892.png', '15256.png', '15800.png', '09261.png', '08672.png', '00143.png', '07828.png', '23984.png', '07647.png', '19486.png', '23797.png', '07139.png', '23544.png', '12296.png', '17545.png', '20076.png', '00571.png', '03425.png', '11082.png', '18615.png', '14256.png', '02885.png', '21079.png', '02452.png', '23561.png', '10114.png', '16145.png', '10413.png', '23259.png', '11890.png', '22416.png', '04853.png', '13533.png', '14593.png', '07927.png', '08762.png', '23607.png', '09374.png', '21008.png', '22178.png', '09682.png', '08859.png', '04351.png', '00153.png', '06238.png', '07814.png', '07536.png', '00185.png', '13344.png', '00829.png', '22722.png', '10949.png', '15227.png', '16287.png', '11673.png', '07153.png', '22620.png', '11336.png', '14194.png', '04970.png', '04649.png', '10954.png', '13361.png', '19087.png', '20695.png', '24548.png', '09858.png', '18219.png', '16335.png', '18494.png', '08719.png', '24166.png', '17896.png', '22541.png', '01770.png', '11766.png', '11731.png', '15091.png', '13862.png', '24466.png', '02028.png', '10471.png', '06748.png', '19177.png', '19294.png', '06729.png', '15778.png', '02020.png', '15585.png', '02623.png', '21180.png', '09479.png', '06491.png', '23240.png', '05720.png', '02567.png', '01022.png', '11109.png', '05479.png', '24534.png', '15364.png', '15978.png', '16192.png', '05932.png', '04153.png', '22981.png', '04667.png', '17493.png', '16855.png', '18403.png', '00442.png', '20667.png', '21166.png', '23132.png', '12030.png', '24207.png', '20993.png', '20884.png', '01387.png', '08760.png', '22064.png', '18555.png', '13870.png', '11698.png', '06076.png', '13990.png', '21921.png', '08968.png', '06218.png', '06035.png', '09799.png', '10644.png', '14295.png', '11567.png', '16447.png', '03490.png', '14102.png', '05958.png', '10778.png', '02921.png', '24704.png', '17150.png', '23468.png', '14253.png', '11112.png', '17954.png', '06354.png', '07870.png', '20256.png', '16134.png', '18409.png', '02886.png', '11232.png', '17341.png', '09806.png', '21591.png', '23900.png', '02704.png', '09397.png', '18717.png', '13678.png', '19388.png', '21149.png', '19579.png', '00973.png', '05362.png', '08834.png', '03650.png', '04229.png', '16585.png', '08043.png', '00235.png', '09733.png', '15110.png', '04708.png', '23239.png', '15508.png', '15447.png', '03935.png', '10925.png', '12244.png', '06274.png', '13056.png', '07909.png', '08935.png', '22634.png', '02185.png', '23069.png', '18163.png', '10713.png', '15933.png', '03983.png', '19338.png', '03719.png', '13302.png', '23964.png', '18689.png', '12322.png', '22468.png', '05373.png', '05395.png', '07140.png', '13592.png', '09953.png', '21391.png', '17983.png', '04965.png', '17460.png', '15687.png', '21527.png', '21076.png', '10337.png', '13283.png', '10599.png', '17764.png', '20535.png', '15924.png', '12410.png', '09253.png', '15367.png', '13435.png', '22101.png', '18850.png', '12530.png', '13559.png', '03271.png', '00948.png', '11030.png', '04477.png', '11316.png', '19506.png', '23982.png', '21976.png', '06390.png', '05131.png', '07348.png', '00629.png', '14101.png', '15092.png', '00945.png', '14787.png', '11945.png', '19070.png', '17408.png', '09420.png', '23531.png', '03962.png', '09013.png', '01484.png', '04982.png', '21432.png', '23436.png', '22157.png', '20692.png', '18080.png', '16469.png', '07837.png', '24533.png', '20214.png', '03587.png', '04019.png', '03644.png', '01239.png', '21670.png', '21892.png', '24505.png', '16143.png', '12564.png', '18527.png', '21226.png', '15379.png', '02339.png', '21273.png', '12455.png', '18659.png', '02370.png', '24275.png', '03398.png', '16304.png', '17882.png', '06928.png', '22449.png', '24699.png', '11356.png', '05091.png', '24513.png', '17974.png', '14168.png', '07395.png', '00576.png', '11790.png', '22501.png', '11650.png', '23413.png', '17127.png', '20985.png', '00622.png', '24361.png', '24135.png', '20901.png', '03397.png', '12481.png', '02904.png', '01112.png', '06171.png', '14248.png', '11322.png', '09172.png', '04113.png', '19889.png', '13938.png', '06621.png', '10005.png', '06478.png', '11307.png', '23202.png', '05006.png', '21898.png', '18209.png', '15179.png', '15405.png', '11732.png', '10203.png', '12813.png', '09578.png', '24126.png', '20350.png', '07661.png', '16091.png', '08552.png', '04234.png', '13172.png', '22761.png', '22025.png', '18882.png', '00127.png', '13049.png', '21533.png', '18785.png', '13794.png', '10165.png', '17753.png', '10582.png', '11451.png', '02325.png', '05146.png', '11211.png', '21634.png', '08924.png', '09123.png', '14060.png', '06766.png', '05567.png', '06796.png', '20591.png', '14686.png', '16766.png', '03025.png', '04404.png', '19601.png', '02162.png', '23576.png', '03099.png', '07607.png', '07040.png', '23992.png', '08258.png', '05438.png', '03755.png', '15610.png', '19589.png', '08805.png', '20254.png', '07764.png', '17515.png', '01133.png', '18020.png', '13549.png', '22859.png', '12116.png', '02777.png', '23500.png', '19745.png', '08715.png', '21522.png', '14466.png', '14527.png', '19675.png', '07548.png', '01968.png', '21212.png', '09791.png', '10909.png', '21269.png', '03757.png', '16144.png', '16587.png', '13903.png', '06077.png', '09835.png', '06689.png', '05305.png', '08833.png', '06249.png', '03160.png', '05550.png', '03898.png', '04867.png', '03214.png', '08782.png', '02766.png', '07280.png', '23600.png', '06757.png', '13423.png', '20307.png', '05843.png', '01101.png', '01012.png', '11627.png', '19997.png', '22054.png', '06588.png', '01857.png', '03969.png', '18554.png', '03566.png', '02849.png', '07688.png', '14986.png', '03041.png', '06825.png', '24954.png', '23771.png', '07622.png', '03815.png', '05066.png', '10514.png', '14954.png', '06596.png', '21057.png', '24014.png', '13612.png', '06389.png', '14866.png', '14685.png', '12958.png', '21652.png', '06663.png', '02004.png', '21982.png', '17376.png', '13838.png', '00733.png', '11495.png', '02638.png', '13152.png', '23670.png', '09203.png', '20125.png', '02819.png', '21266.png', '19694.png', '23847.png', '24421.png', '09375.png', '16517.png', '11977.png', '04685.png', '08951.png', '01831.png', '14348.png', '15722.png', '04510.png', '01137.png', '12387.png', '19377.png', '01441.png', '13199.png', '16749.png', '04801.png', '02050.png', '22114.png', '04808.png', '01907.png', '18772.png', '08082.png', '07602.png', '08826.png', '18736.png', '20086.png', '16078.png', '13092.png', '10171.png', '01469.png', '15626.png', '22992.png', '08856.png', '03017.png', '13097.png', '17964.png', '08459.png', '09132.png', '13765.png', '22802.png', '10395.png', '24530.png', '03159.png', '19209.png', '09237.png', '22744.png', '05203.png', '16273.png', '08392.png', '23152.png', '22042.png', '19155.png', '04462.png', '20332.png', '23098.png', '19809.png', '19415.png', '02070.png', '15392.png', '10428.png', '22482.png', '17628.png', '01475.png', '24858.png', '12212.png', '13400.png', '08605.png', '13315.png', '03668.png', '12524.png', '12407.png', '00966.png', '10950.png', '14230.png', '11192.png', '04456.png', '12889.png', '02357.png', '01123.png', '00761.png', '20436.png', '13426.png', '09966.png', '13529.png', '11136.png', '12329.png', '19678.png', '06741.png', '02783.png', '14760.png', '13456.png', '14165.png', '04299.png', '06051.png', '23491.png', '05964.png', '03442.png', '18567.png', '13209.png', '22785.png', '14748.png', '07190.png', '04347.png', '19941.png', '22513.png', '04802.png', '20120.png', '11591.png', '16170.png', '04445.png', '13190.png', '06347.png', '23904.png', '22915.png', '10841.png', '03676.png', '11333.png', '05625.png', '00199.png', '09436.png', '23284.png', '17092.png', '22780.png', '11517.png', '09545.png', '12967.png', '02133.png', '15170.png', '00870.png', '23509.png', '18205.png', '23776.png', '16280.png', '21579.png', '06610.png', '19719.png', '18997.png', '20409.png', '08794.png', '05811.png', '01250.png', '23031.png', '11929.png', '12762.png', '21344.png', '02203.png', '21975.png', '12121.png', '01329.png', '04188.png', '21939.png', '18261.png', '22280.png', '18078.png', '23535.png', '11183.png', '03746.png', '23865.png', '21470.png', '22499.png', '22675.png', '13024.png', '02163.png', '06959.png', '02121.png', '06192.png', '06967.png', '13834.png', '16261.png', '17137.png', '21358.png', '23155.png', '05309.png', '14816.png', '19904.png', '06717.png', '21261.png', '22012.png', '21667.png', '01815.png', '22725.png', '21264.png', '19736.png', '07282.png', '11634.png', '03660.png', '13350.png', '14789.png', '15651.png', '15607.png', '08637.png', '13096.png', '08248.png', '22958.png', '04718.png', '18128.png', '15807.png', '24047.png', '17563.png', '21473.png', '22703.png', '13047.png', '03347.png', '17228.png', '05399.png', '09484.png', '01536.png', '05677.png', '19190.png', '08046.png', '21021.png', '10044.png', '06434.png', '03089.png', '14392.png', '02539.png', '23751.png', '05588.png', '00415.png', '03306.png', '18753.png', '10697.png', '00241.png', '19853.png', '11286.png', '08572.png', '00345.png', '19002.png', '01421.png', '19224.png', '09909.png', '18697.png', '09582.png', '10200.png', '13357.png', '15855.png', '22420.png', '02304.png', '10639.png', '12635.png', '19222.png', '12666.png', '16166.png', '07913.png', '09457.png', '03685.png', '19326.png', '23422.png', '04215.png', '16779.png', '15218.png', '11712.png', '10733.png', '16055.png', '08423.png', '18126.png', '03057.png', '20308.png', '23542.png', '18522.png', '20361.png', '05807.png', '13101.png', '03027.png', '01494.png', '09847.png', '04538.png', '04668.png', '07476.png', '23558.png', '13513.png', '02223.png', '14314.png', '16558.png', '21566.png', '15430.png', '16904.png', '20408.png', '08067.png', '06012.png', '13029.png', '14744.png', '00156.png', '19693.png', '17404.png', '24301.png', '09938.png', '05465.png', '24468.png', '22241.png', '02799.png', '11574.png', '15631.png', '04248.png', '21729.png', '08199.png', '13571.png', '15603.png', '20982.png', '11092.png', '05993.png', '01433.png', '12232.png', '13040.png', '06803.png', '01426.png', '06339.png', '16733.png', '03856.png', '08345.png', '08108.png', '17204.png', '09018.png', '21632.png', '23725.png', '20316.png', '11468.png', '19943.png', '10581.png', '15743.png', '19639.png', '17510.png', '20043.png', '12194.png', '19720.png', '06692.png', '18892.png', '07867.png', '06512.png', '21347.png', '15400.png', '20590.png', '07109.png', '24494.png', '03867.png', '03598.png', '09340.png', '18250.png', '21741.png', '23469.png', '08816.png', '05433.png', '07902.png', '08221.png', '17126.png', '09277.png', '12391.png', '22053.png', '23154.png', '18227.png', '19861.png', '06206.png', '24804.png', '24643.png', '16378.png', '02323.png', '06581.png', '16578.png', '08329.png', '24253.png', '24279.png', '05026.png', '19800.png', '14387.png', '23390.png', '03278.png', '01854.png', '15793.png', '03852.png', '22904.png', '18795.png', '03338.png', '11266.png', '22089.png', '08728.png', '06545.png', '16978.png', '12597.png', '03312.png', '14481.png', '04981.png', '02564.png', '17476.png', '21574.png', '24634.png', '14776.png', '10591.png', '10248.png', '21216.png', '07682.png', '16951.png', '21448.png', '16544.png', '07772.png', '04559.png', '17572.png', '04207.png', '23612.png', '19916.png', '15445.png', '05261.png', '11093.png', '19919.png', '17188.png', '06794.png', '01663.png', '09164.png', '19625.png', '22833.png', '10842.png', '17271.png', '08771.png', '01084.png', '10202.png', '06733.png', '22287.png', '20922.png', '08803.png', '04564.png', '17436.png', '06957.png', '10340.png', '15577.png', '19004.png', '02073.png', '21440.png', '01001.png', '07334.png', '03828.png', '19182.png', '01596.png', '04996.png', '16666.png', '07365.png', '08853.png', '16505.png', '09151.png', '07519.png', '12085.png', '08061.png', '03854.png', '14250.png', '06998.png', '23826.png', '21814.png', '13966.png', '09101.png', '20732.png', '02748.png', '16230.png', '11733.png', '06838.png', '03020.png', '17415.png', '20464.png', '10379.png', '01303.png', '05178.png', '24445.png', '16830.png', '07372.png', '01825.png', '17311.png', '01645.png', '00268.png', '13563.png', '22977.png', '22745.png', '19502.png', '18809.png', '07189.png', '02463.png', '16871.png', '05328.png', '20814.png', '08285.png', '18874.png', '18277.png', '13462.png', '09086.png', '05996.png', '08567.png', '01876.png', '09600.png', '20519.png', '11704.png', '14449.png', '11719.png', '23402.png', '13748.png', '18436.png', '22103.png', '12035.png', '17951.png', '05284.png', '19689.png', '06258.png', '11478.png', '00441.png', '18694.png', '10984.png', '15093.png', '23410.png', '21804.png', '23241.png', '21024.png', '05412.png', '18404.png', '16094.png', '07037.png', '02519.png', '13449.png', '07215.png', '07522.png', '15733.png', '20200.png', '12575.png', '20529.png', '08062.png', '17513.png', '07715.png', '04394.png', '02902.png', '09810.png', '13345.png', '23471.png', '14836.png', '01987.png', '24777.png', '06813.png', '14219.png', '20061.png', '07053.png', '20958.png', '21220.png', '14889.png', '03464.png', '22014.png', '17138.png', '06437.png', '05497.png', '21911.png', '21720.png', '20731.png', '23837.png', '24371.png', '10879.png', '12113.png', '16121.png', '07410.png', '06850.png', '02758.png', '16177.png', '00790.png', '16110.png', '10801.png', '15123.png', '08984.png', '24520.png', '13792.png', '12796.png', '12383.png', '13910.png', '09980.png', '24511.png', '22818.png', '11828.png', '18654.png', '17465.png', '18501.png', '16459.png', '01037.png', '12034.png', '10573.png', '02963.png', '19863.png', '17293.png', '11723.png', '12622.png', '12025.png', '19238.png', '11382.png', '11957.png', '12713.png', '06078.png', '13787.png', '12225.png', '24244.png', '10910.png', '09952.png', '22211.png', '07238.png', '09360.png', '01635.png', '07201.png', '24040.png', '10722.png', '11813.png', '17143.png', '05087.png', '21876.png', '07632.png', '06854.png', '12812.png', '23095.png', '02117.png', '16972.png', '03383.png', '21055.png', '03472.png', '08002.png', '21534.png', '16509.png', '02177.png', '04461.png', '00683.png', '05102.png', '10543.png', '17196.png', '03816.png', '21426.png', '06102.png', '23793.png', '05082.png', '24160.png', '20672.png', '05473.png', '18526.png', '12122.png', '22494.png', '15857.png', '20156.png', '24481.png', '11998.png', '00524.png', '16838.png', '09789.png', '09111.png', '20952.png', '10911.png', '00340.png', '18889.png', '22362.png', '13454.png', '03228.png', '02084.png', '08720.png', '15898.png', '11473.png', '00208.png', '18626.png', '04260.png', '21256.png', '10102.png', '03963.png', '16239.png', '01964.png', '07304.png', '17500.png', '09767.png', '14981.png', '22082.png', '17297.png', '05198.png', '06918.png', '02713.png', '09134.png', '05013.png', '07589.png', '18121.png', '23074.png', '20636.png', '00420.png', '11480.png', '10975.png', '06237.png', '21994.png', '14608.png', '11297.png', '24429.png', '07671.png', '10184.png', '07588.png', '23093.png', '18723.png', '11387.png', '22233.png', '11345.png', '23595.png', '12444.png', '13419.png', '13038.png', '08104.png', '14352.png', '06524.png', '02407.png', '20665.png', '15277.png', '07524.png', '06740.png', '09259.png', '18338.png', '08536.png', '07486.png', '05957.png', '22254.png', '10418.png', '17503.png', '20401.png', '21583.png', '24810.png', '02491.png', '23976.png', '05150.png', '04414.png', '01691.png', '19556.png', '02689.png', '17051.png', '16391.png', '22364.png', '20808.png', '00685.png', '11988.png', '21355.png', '03834.png', '00091.png', '00133.png', '03997.png', '11891.png', '12449.png', '24248.png', '16281.png', '04793.png', '04604.png', '15684.png', '04734.png', '09249.png', '01292.png', '19542.png', '19066.png', '01875.png', '02784.png', '18812.png', '09406.png', '13890.png', '02781.png', '17466.png', '21954.png', '20222.png', '06746.png', '22350.png', '13491.png', '21981.png', '21108.png', '17687.png', '18709.png', '19735.png', '11444.png', '17506.png', '22997.png', '22412.png', '18661.png', '07148.png', '15591.png', '18331.png', '00672.png', '09754.png', '03274.png', '02354.png', '22256.png', '06414.png', '09843.png', '08832.png', '19704.png', '01015.png', '23466.png', '02401.png', '01999.png', '11219.png', '22836.png', '00914.png', '04917.png', '13820.png', '09023.png', '00092.png', '19057.png', '07458.png', '19113.png', '21754.png', '06971.png', '18229.png', '04908.png', '10847.png', '24738.png', '21961.png', '19772.png', '06694.png', '18370.png', '11099.png', '12335.png', '20337.png', '18990.png', '22528.png', '13769.png', '23038.png', '10802.png', '23448.png', '18321.png', '08149.png', '02494.png', '13227.png', '10258.png', '05387.png', '22635.png', '14581.png', '21334.png', '05558.png', '14693.png', '20049.png', '07893.png', '02680.png', '09900.png', '08944.png', '14770.png', '05138.png', '00875.png', '04682.png', '13594.png', '14450.png', '17522.png', '05274.png', '11547.png', '03987.png', '13699.png', '07096.png', '08633.png', '23613.png', '11615.png', '24690.png', '01582.png', '14157.png', '10111.png', '06586.png', '01097.png', '21854.png', '01690.png', '11220.png', '17179.png', '14020.png', '09260.png', '21238.png', '10075.png', '02445.png', '19166.png', '12721.png', '22119.png', '16418.png', '20788.png', '17442.png', '11734.png', '11115.png', '23032.png', '23661.png', '24397.png', '17957.png', '19865.png', '12675.png', '11411.png', '16383.png', '00546.png', '00201.png', '24532.png', '07512.png', '15742.png', '05730.png', '24103.png', '02626.png', '22694.png', '18629.png', '02204.png', '23203.png', '06656.png', '03632.png', '21438.png', '06948.png', '03058.png', '09640.png', '07079.png', '00640.png', '15958.png', '20139.png', '14768.png', '10135.png', '18796.png', '23023.png', '20673.png', '08168.png', '08206.png', '13360.png', '14186.png', '19499.png', '10399.png', '01868.png', '01058.png', '23570.png', '16337.png', '16165.png', '07988.png', '18009.png', '01358.png', '13131.png', '06160.png', '04227.png', '22079.png', '01190.png', '21959.png', '04595.png', '08665.png', '10003.png', '01882.png', '13171.png', '04594.png', '21317.png', '13604.png', '21935.png', '16522.png', '01086.png', '17160.png', '05389.png', '12599.png', '11108.png', '19850.png', '04218.png', '15789.png', '16088.png', '11234.png', '14315.png', '20182.png', '06075.png', '03822.png', '14452.png', '12378.png', '20009.png', '02102.png', '07141.png', '06091.png', '16208.png', '00894.png', '23426.png', '04729.png', '06704.png', '02915.png', '16073.png', '12682.png', '13178.png', '14533.png', '13935.png', '11600.png', '04162.png', '08765.png', '23124.png', '16178.png', '09992.png', '24613.png', '11806.png', '23528.png', '12185.png', '15821.png', '24755.png', '09170.png', '17170.png', '23269.png', '04489.png', '00816.png', '06082.png', '08916.png', '21491.png', '20986.png', '19797.png', '00519.png', '07439.png', '17164.png', '08651.png', '01997.png', '23582.png', '20777.png', '23924.png', '20324.png', '22209.png', '04342.png', '15391.png', '18132.png', '11106.png', '07117.png', '03280.png', '04058.png', '15018.png', '05651.png', '02534.png', '06759.png', '11433.png', '10690.png', '07737.png', '11422.png', '21349.png', '01919.png', '12705.png', '08746.png', '18214.png', '14739.png', '02444.png', '16138.png', '23652.png', '06831.png', '01524.png', '24414.png', '17390.png', '17206.png', '08621.png', '07445.png', '14028.png', '10820.png', '14363.png', '15771.png', '05985.png', '15166.png', '23949.png', '14227.png', '10827.png', '15822.png', '17490.png', '23507.png', '22561.png', '15249.png', '07310.png', '10548.png', '08100.png', '12811.png', '17639.png', '18917.png', '01733.png', '01305.png', '06150.png', '14975.png', '12513.png', '11210.png', '21142.png', '17859.png', '15210.png', '21849.png', '16006.png', '08540.png', '07801.png', '19585.png', '17804.png', '17892.png', '15775.png', '17411.png', '24603.png', '16593.png', '07089.png', '06516.png', '04553.png', '09089.png', '14651.png', '24004.png', '23869.png', '03355.png', '01742.png', '22976.png', '13242.png', '16265.png', '16709.png', '00012.png', '04806.png', '08319.png', '12079.png', '04482.png', '06080.png', '10043.png', '10266.png', '07301.png', '22215.png', '22339.png', '13427.png', '06385.png', '17919.png', '02484.png', '04236.png', '09964.png', '00207.png', '04572.png', '20626.png', '14586.png', '08595.png', '14456.png', '07110.png', '09234.png', '15889.png', '10387.png', '16956.png', '10132.png', '16556.png', '07431.png', '13864.png', '07271.png', '23306.png', '19777.png', '23925.png', '18539.png', '02471.png', '06045.png', '11375.png', '06747.png', '14051.png', '19353.png', '19813.png', '01830.png', '06687.png', '16161.png', '22125.png', '00152.png', '09001.png', '05917.png', '08291.png', '06282.png', '00131.png', '23810.png', '21562.png', '04163.png', '05771.png', '16428.png', '04457.png', '23315.png', '20864.png', '19455.png', '21978.png', '23488.png', '16611.png', '02376.png', '15602.png', '15627.png', '14741.png', '02512.png', '11990.png', '24365.png', '15135.png', '11744.png', '10481.png', '03109.png', '11360.png', '05371.png', '24700.png', '10931.png', '10090.png', '16594.png', '03871.png', '10012.png', '19245.png', '00355.png', '02619.png', '10618.png', '09443.png', '14630.png', '13566.png', '24711.png', '13879.png', '01375.png', '05028.png', '22147.png', '08702.png', '15406.png', '09619.png', '02291.png', '23421.png', '16550.png', '10933.png', '16844.png', '22970.png', '00926.png', '11816.png', '00993.png', '17431.png', '14908.png', '15550.png', '03356.png', '01818.png', '12114.png', '16331.png', '06504.png', '08166.png', '17825.png', '00310.png', '01912.png', '13395.png', '17866.png', '21812.png', '02501.png', '09312.png', '11901.png', '24224.png', '16926.png', '10706.png', '19069.png', '13681.png', '17641.png', '24308.png', '21096.png', '13835.png', '06833.png', '05281.png', '01299.png', '21336.png', '05531.png', '20241.png', '05124.png', '01636.png', '05245.png', '07233.png', '05080.png', '17940.png', '08105.png', '18864.png', '15702.png', '07435.png', '18663.png', '14875.png', '01989.png', '03966.png', '16974.png', '18266.png', '02544.png', '18085.png', '09740.png', '20755.png', '09689.png', '14877.png', '23667.png', '08893.png', '04807.png', '12582.png', '03404.png', '14355.png', '01184.png', '09099.png', '07093.png', '05404.png', '02061.png', '16730.png', '05810.png', '05201.png', '08772.png', '17759.png', '04539.png', '08513.png', '16561.png', '22033.png', '13325.png', '10503.png', '14655.png', '21195.png', '20011.png', '02486.png', '13043.png', '10326.png', '08610.png', '07873.png', '20262.png', '18208.png', '00644.png', '23071.png', '02207.png', '12905.png', '06515.png', '08678.png', '24347.png', '22196.png', '07725.png', '12645.png', '22579.png', '12362.png', '00792.png', '00754.png', '14289.png', '22648.png', '13493.png', '20730.png', '00764.png', '17287.png', '08627.png', '21677.png', '18571.png', '14767.png', '19319.png', '17568.png', '15421.png', '20978.png', '18873.png', '02634.png', '14642.png', '06287.png', '19711.png', '03118.png', '23046.png', '14147.png', '08525.png', '05660.png', '11040.png', '24251.png', '23683.png', '14937.png', '17897.png', '17487.png', '24465.png', '19776.png', '18355.png', '12985.png', '12540.png', '09231.png', '02458.png', '19762.png', '20023.png', '23485.png', '02794.png', '18863.png', '24124.png', '10585.png', '10155.png', '09755.png', '07749.png', '16467.png', '24277.png', '07898.png', '03104.png', '00823.png', '08862.png', '06876.png', '14188.png', '03633.png', '13039.png', '08352.png', '14946.png', '02187.png', '04557.png', '00715.png', '08685.png', '20782.png', '16253.png', '19875.png', '11448.png', '21467.png', '22124.png', '06349.png', '12478.png', '04409.png', '14848.png', '04618.png', '00759.png', '03305.png', '09045.png', '17566.png', '23969.png', '03751.png', '16652.png', '11925.png', '11849.png', '22023.png', '18390.png', '00559.png', '20759.png', '12055.png', '19492.png', '20632.png', '24369.png', '09628.png', '03695.png', '12994.png', '20514.png', '02402.png', '20949.png', '08800.png', '19131.png', '22373.png', '22670.png', '17942.png', '17921.png', '14793.png', '16066.png', '02247.png', '14607.png', '00144.png', '11391.png', '06941.png', '15278.png', '12799.png', '16679.png', '09320.png', '00291.png', '02419.png', '20452.png', '12208.png', '07397.png', '11775.png', '10307.png', '01373.png', '11265.png', '00730.png', '13262.png', '23158.png', '14959.png', '16644.png', '01655.png', '23540.png', '24012.png', '12521.png', '10180.png', '09258.png', '09522.png', '23440.png', '08852.png', '10476.png', '04582.png', '02058.png', '22863.png', '10265.png', '22450.png', '19848.png', '03818.png', '21509.png', '01920.png', '05397.png', '04368.png', '19284.png', '24245.png', '21205.png', '17862.png', '08512.png', '16083.png', '13210.png', '16172.png', '09787.png', '09005.png', '01073.png', '03003.png', '01945.png', '10914.png', '20894.png', '07537.png', '19507.png', '05175.png', '04656.png', '07776.png', '10604.png', '02071.png', '18959.png', '19535.png', '17592.png', '10328.png', '18378.png', '09575.png', '10218.png', '12248.png', '16257.png', '02918.png', '17446.png', '12596.png', '18817.png', '05897.png', '06252.png', '00302.png', '16400.png', '11317.png', '04620.png', '10572.png', '05809.png', '07308.png', '07592.png', '08208.png', '08305.png', '18420.png', '10374.png', '06849.png', '15373.png', '08963.png', '07059.png', '11458.png', '04443.png', '05977.png', '21025.png', '07818.png', '19915.png', '16574.png', '08087.png', '11165.png', '09039.png', '00949.png', '14114.png', '14912.png', '14049.png', '07775.png', '18511.png', '00930.png', '13484.png', '15867.png', '15226.png', '21721.png', '06639.png', '22081.png', '04398.png', '05646.png', '23567.png', '19198.png', '00467.png', '02674.png', '08740.png', '08476.png', '13762.png', '04745.png', '18637.png', '03954.png', '00805.png', '08758.png', '06706.png', '20654.png', '06503.png', '09470.png', '04691.png', '04859.png', '13615.png', '00523.png', '20729.png', '22487.png', '22658.png', '00880.png', '05717.png', '10488.png', '13636.png', '17543.png', '24839.png', '21321.png', '12863.png', '06806.png', '18488.png', '14100.png', '00421.png', '15594.png', '23182.png', '14132.png', '01796.png', '17042.png', '10934.png', '11541.png', '07822.png', '18137.png', '14864.png', '07064.png', '02390.png', '00923.png', '14925.png', '19463.png', '06179.png', '18015.png', '14112.png', '08461.png', '02236.png', '14994.png', '15645.png', '03476.png', '20384.png', '10123.png', '22569.png', '05718.png', '12857.png', '14736.png', '06475.png', '20799.png', '12600.png', '16489.png', '18456.png', '15117.png', '01623.png', '22407.png', '12005.png', '03965.png', '10789.png', '19971.png', '20623.png', '01430.png', '13589.png', '24379.png', '15913.png', '19949.png', '13083.png', '24214.png', '05116.png', '13295.png', '13187.png', '00380.png', '21871.png', '05068.png', '08409.png', '07922.png', '09209.png', '10431.png', '06921.png', '06832.png', '18352.png', '00095.png', '05908.png', '18313.png', '08674.png', '00077.png', '03391.png', '16320.png', '03651.png', '20209.png', '14374.png', '23498.png', '21335.png', '14841.png', '13888.png', '10711.png', '14948.png', '16147.png', '01614.png', '20666.png', '12291.png', '03993.png', '10403.png', '15426.png', '09027.png', '18973.png', '22138.png', '21577.png', '03958.png', '04374.png', '20754.png', '18305.png', '02786.png', '18915.png', '19138.png', '17036.png', '10439.png', '15801.png', '17253.png', '21181.png', '19903.png', '03324.png', '06738.png', '13323.png', '07722.png', '21803.png', '17646.png', '11226.png', '18926.png', '19021.png', '21367.png', '19668.png', '01011.png', '08989.png', '13141.png', '13916.png', '06426.png', '16555.png', '20581.png', '11032.png', '04164.png', '17688.png', '03095.png', '24658.png', '00424.png', '22383.png', '01196.png', '10073.png', '22827.png', '07225.png', '00639.png', '01185.png', '08764.png', '14499.png', '03968.png', '07482.png', '16801.png', '05104.png', '02557.png', '02182.png', '02011.png', '06792.png', '18569.png', '21731.png', '18953.png', '23039.png', '05388.png', '00200.png', '02706.png', '06225.png', '02322.png', '15971.png', '14783.png', '10542.png', '23838.png', '22638.png', '10791.png', '18345.png', '15105.png', '14444.png', '06129.png', '17402.png', '10060.png', '22058.png', '20549.png', '10642.png', '05914.png', '24966.png', '00689.png', '01527.png', '23881.png', '24198.png', '21772.png', '00739.png', '00877.png', '17877.png', '21852.png', '13204.png', '18671.png', '11505.png', '04654.png', '21538.png', '08863.png', '19874.png', '09537.png', '23692.png', '13983.png', '09557.png', '24938.png', '19092.png', '03730.png', '05378.png', '00098.png', '06784.png', '16164.png', '15728.png', '14933.png', '20576.png', '04823.png', '18086.png', '24456.png', '20085.png', '03669.png', '08047.png', '18908.png', '12750.png', '06211.png', '08688.png', '08921.png', '01253.png', '11134.png', '00588.png', '24350.png', '20135.png', '14615.png', '08649.png', '22234.png', '03532.png', '18600.png', '18702.png', '20195.png', '21960.png', '05228.png', '16889.png', '15691.png', '19366.png', '06448.png', '18241.png', '06585.png', '04744.png', '04739.png', '09426.png', '22989.png', '08196.png', '01434.png', '01235.png', '06700.png', '01602.png', '17713.png', '20718.png', '13767.png', '21156.png', '24474.png', '03697.png', '00162.png', '05379.png', '06975.png', '05865.png', '05472.png', '13943.png', '08463.png', '08615.png', '21002.png', '07964.png', '15994.png', '03196.png', '09943.png', '08256.png', '09757.png', '21997.png', '07480.png', '19079.png', '03802.png', '00874.png', '09303.png', '23502.png', '15128.png', '21027.png', '07496.png', '24859.png', '00618.png', '17403.png', '17517.png', '24316.png', '24628.png', '13474.png', '00706.png', '07088.png', '19437.png', '08010.png', '01571.png', '10268.png', '11690.png', '06013.png', '10745.png', '14364.png', '02679.png', '06715.png', '10875.png', '16821.png', '05333.png', '01862.png', '06332.png', '09546.png', '16892.png', '24147.png', '13010.png', '14089.png', '01234.png', '18621.png', '09033.png', '06029.png', '15285.png', '19868.png', '00005.png', '16807.png', '05093.png', '02548.png', '16893.png', '20580.png', '06166.png', '09244.png', '01391.png', '21265.png', '07050.png', '07809.png', '07279.png', '08364.png', '08703.png', '05579.png', '09063.png', '09970.png', '05967.png', '09098.png', '07116.png', '10588.png', '21602.png', '23434.png', '24083.png', '05912.png', '12944.png', '00330.png', '00687.png', '14104.png', '00853.png', '19094.png', '09813.png', '16213.png', '05074.png', '17247.png', '09669.png', '16324.png', '10104.png', '14460.png', '23105.png', '00668.png', '03049.png', '16466.png', '17497.png', '02900.png', '10805.png', '00928.png', '03210.png', '20896.png', '10876.png', '21048.png', '19317.png', '09914.png', '15777.png', '14006.png', '24592.png', '09884.png', '03023.png', '23395.png', '10404.png', '16628.png', '09667.png', '00483.png', '13415.png', '17107.png', '19615.png', '02065.png', '22774.png', '04027.png', '18028.png', '21417.png', '11222.png', '01633.png', '02788.png', '21442.png', '00352.png', '04543.png', '01318.png', '20060.png', '00508.png', '21372.png', '22013.png', '14149.png', '21272.png', '14272.png', '05331.png', '21088.png', '12551.png', '02975.png', '08960.png', '21481.png', '24703.png', '16803.png', '11748.png', '05934.png', '06205.png', '00694.png', '04259.png', '06223.png', '00767.png', '06017.png', '22973.png', '24053.png', '05627.png', '09472.png', '14014.png', '12948.png', '06320.png', '06334.png', '03111.png', '00119.png', '15487.png', '23685.png', '18407.png', '21964.png', '16717.png', '14978.png', '09818.png', '06634.png', '23389.png', '11068.png', '08371.png', '13711.png', '00182.png', '24509.png', '14260.png', '00477.png', '19753.png', '15237.png', '00745.png', '04575.png', '22174.png', '20606.png', '22946.png', '10530.png', '07960.png', '18638.png', '08716.png', '09344.png', '13469.png', '12788.png', '01775.png', '19633.png', '01988.png', '12151.png', '02036.png', '00075.png', '11780.png', '07133.png', '23873.png', '21418.png', '11045.png', '09129.png', '07495.png', '02897.png', '12061.png', '23168.png', '23727.png', '19293.png', '02120.png', '16387.png', '00198.png', '03923.png', '01873.png', '18401.png', '04789.png', '20557.png', '24435.png', '18937.png', '23099.png', '03297.png', '24032.png', '16024.png', '14235.png', '16030.png', '01829.png', '01641.png', '15155.png', '08324.png', '02661.png', '10223.png', '04952.png', '18413.png', '18834.png', '02901.png', '24142.png', '07732.png', '23106.png', '11489.png', '21510.png', '16382.png', '04397.png', '14702.png', '21588.png', '12139.png', '07986.png', '02592.png', '12660.png', '18510.png', '24668.png', '18871.png', '24169.png', '03862.png', '23195.png', '24206.png', '22693.png', '11993.png', '10919.png', '09021.png', '09345.png', '18464.png', '12255.png', '10217.png', '17529.png', '16130.png', '04511.png', '15598.png', '21316.png', '18451.png', '10963.png', '24263.png', '01501.png', '23350.png', '18478.png', '14286.png', '00974.png', '10614.png', '03803.png', '14223.png', '22484.png', '06613.png', '11856.png', '17784.png', '14529.png', '21828.png', '01813.png', '15409.png', '20315.png', '02466.png', '01529.png', '15012.png', '08749.png', '01060.png', '16911.png', '03679.png', '01675.png', '23653.png', '03276.png', '12233.png', '24553.png', '11022.png', '05937.png', '13755.png', '06725.png', '07990.png', '09053.png', '11875.png', '18673.png', '15313.png', '24723.png', '02381.png', '20119.png', '15196.png', '06736.png', '14638.png', '01760.png', '05264.png', '07033.png', '15515.png', '04285.png', '17920.png', '18691.png', '24631.png', '01448.png', '11484.png', '00593.png', '03249.png', '14677.png', '09006.png', '22201.png', '20948.png', '19553.png', '05952.png', '17290.png', '18443.png', '16022.png', '24270.png', '24037.png', '03902.png', '00072.png', '21948.png', '20039.png', '08253.png', '10970.png', '20766.png', '16290.png', '16156.png', '18284.png', '16291.png', '18742.png', '02474.png', '05105.png', '18239.png', '01135.png', '21032.png', '07947.png', '03071.png', '05705.png', '15770.png', '07106.png', '21614.png', '08279.png', '04955.png', '08165.png', '18616.png', '01984.png', '19412.png', '00169.png', '15528.png', '03664.png', '24372.png', '11272.png', '02832.png', '20223.png', '23342.png', '17824.png', '16366.png', '02867.png', '12933.png', '06781.png', '17585.png', '00240.png', '23481.png', '20257.png', '15095.png', '05255.png', '12126.png', '03772.png', '12567.png', '09871.png', '06695.png', '05386.png', '18699.png', '10222.png', '03716.png', '22129.png', '02815.png', '17207.png', '01519.png', '20996.png', '19514.png', '06471.png', '19085.png', '11750.png', '14236.png', '09638.png', '14711.png', '11337.png', '23238.png', '02883.png', '05186.png', '24783.png', '17159.png', '03734.png', '09565.png', '03447.png', '16414.png', '07723.png', '10532.png', '07642.png', '14979.png', '14911.png', '17991.png', '01717.png', '06601.png', '18254.png', '18157.png', '19580.png', '16677.png', '17006.png', '12045.png', '18112.png', '14302.png', '13048.png', '12528.png', '19350.png', '21962.png', '18762.png', '02213.png', '09863.png', '03255.png', '03904.png', '20027.png', '00047.png', '14083.png', '05430.png', '10670.png', '00137.png', '09394.png', '08033.png', '19237.png', '04507.png', '09147.png', '10894.png', '15496.png', '12976.png', '17609.png', '05576.png', '17363.png', '00016.png', '15890.png', '21727.png', '08349.png', '17605.png', '24389.png', '00378.png', '04495.png', '07992.png', '01142.png', '06862.png', '14431.png', '20108.png', '14801.png', '21224.png', '20202.png', '09270.png', '04767.png', '07997.png', '20481.png', '17927.png', '07203.png', '15386.png', '24497.png', '18129.png', '10550.png', '08830.png', '10683.png', '15872.png', '20829.png', '16678.png', '07204.png', '09313.png', '14333.png', '04925.png', '03158.png', '00124.png', '21564.png', '22731.png', '09489.png', '09500.png', '00351.png', '04179.png', '17884.png', '12422.png', '00252.png', '07940.png', '17354.png', '23891.png', '07095.png', '00679.png', '07057.png', '07121.png', '08623.png', '01532.png', '20325.png', '10682.png', '03934.png', '03785.png', '17292.png', '05137.png', '14961.png', '21608.png', '02538.png', '04701.png', '21102.png', '14011.png', '12275.png', '00412.png', '11771.png', '03667.png', '20619.png', '21842.png', '18138.png', '24483.png', '07208.png', '08994.png', '02165.png', '17809.png', '22257.png', '10162.png', '18096.png', '22248.png', '17135.png', '18200.png', '09324.png', '10457.png', '22813.png', '06046.png', '13591.png', '24768.png', '22960.png', '18358.png', '05036.png', '22852.png', '12356.png', '03119.png', '12951.png', '22154.png', '08941.png', '11492.png', '02983.png', '20577.png', '16655.png', '00867.png', '24028.png', '18783.png', '00873.png', '03343.png', '13468.png', '19159.png', '08468.png', '05997.png', '01229.png', '19396.png', '15429.png', '08701.png', '06323.png', '23730.png', '11879.png', '05703.png', '18038.png', '05804.png', '06271.png', '12583.png', '22073.png', '18109.png', '03381.png', '18547.png', '07910.png', '12059.png', '11869.png', '11536.png', '11629.png', '11512.png', '24136.png', '11684.png', '09800.png', '13951.png', '14592.png', '07520.png', '06721.png', '04042.png', '13306.png', '16340.png', '07220.png', '03267.png', '03458.png', '16564.png', '06245.png', '02967.png', '19340.png', '20463.png', '21124.png', '10032.png', '04815.png', '20934.png', '10868.png', '08500.png', '05359.png', '13161.png', '02637.png', '17745.png', '24256.png', '13523.png', '14765.png', '00954.png', '08929.png', '05669.png', '20629.png', '19262.png', '21465.png', '07819.png', '22535.png', '10515.png', '15403.png', '09564.png', '01243.png', '16126.png', '07918.png', '19709.png', '04339.png', '08044.png', '12536.png', '23822.png', '10790.png', '05787.png', '18441.png', '06852.png', '04096.png', '24883.png', '04149.png', '14044.png', '20914.png', '15909.png', '19747.png', '22369.png', '09750.png', '08575.png', '17675.png', '12868.png', '01009.png', '22258.png', '16532.png', '20255.png', '12899.png', '03992.png', '08842.png', '11161.png', '12271.png', '14024.png', '16604.png', '23156.png', '06786.png', '21242.png', '10250.png', '11881.png', '20193.png', '03170.png', '14023.png', '14309.png', '00327.png', '09737.png', '02669.png', '02313.png', '22512.png', '12471.png', '22902.png', '01898.png', '08976.png', '15788.png', '19595.png', '16654.png', '06030.png', '18057.png', '11798.png', '15995.png', '13816.png', '23382.png', '23177.png', '07158.png', '14059.png', '09742.png', '22034.png', '20118.png', '00225.png', '12343.png', '24863.png', '21705.png', '24813.png', '18471.png', '22242.png', '20574.png', '04865.png', '14663.png', '15767.png', '14856.png', '20495.png', '06541.png', '09064.png', '10882.png', '04343.png', '15047.png', '00449.png', '13728.png', '09896.png', '06257.png', '04036.png', '24250.png', '21894.png', '15675.png', '14872.png', '21054.png', '05973.png', '02344.png', '04742.png', '03521.png', '11683.png', '11681.png', '10062.png', '12328.png', '08930.png', '12793.png', '23144.png', '21607.png', '07239.png', '03526.png', '21123.png', '08676.png', '07384.png', '20935.png', '07925.png', '05670.png', '19546.png', '19461.png', '02080.png', '02054.png', '02200.png', '05879.png', '24728.png', '01423.png', '21493.png', '10594.png', '23234.png', '10057.png', '19622.png', '14400.png', '04888.png', '03875.png', '03390.png', '04411.png', '24096.png', '17798.png', '03412.png', '15683.png', '17368.png', '18246.png', '22848.png', '07698.png', '09585.png', '18051.png', '07596.png', '21294.png', '21127.png', '11038.png', '16115.png', '04619.png', '15134.png', '18415.png', '23087.png', '07403.png', '05713.png', '19766.png', '02123.png', '11992.png', '12368.png', '01323.png', '15751.png', '23991.png', '02947.png', '03674.png', '07612.png', '17229.png', '07945.png', '00453.png', '20285.png', '11669.png', '09933.png', '19250.png', '13682.png', '06917.png', '02469.png', '18159.png', '04024.png', '22555.png', '08437.png', '11074.png', '04030.png', '12200.png', '03038.png', '18290.png', '11636.png', '05950.png', '00079.png', '22893.png', '20575.png', '24920.png', '09654.png', '02576.png', '05532.png', '22268.png', '02315.png', '03580.png', '19591.png', '19638.png', '22664.png', '24052.png', '13063.png', '12754.png', '07914.png', '23179.png', '19054.png', '12507.png', '04012.png', '19229.png', '11736.png', '12392.png', '22085.png', '07115.png', '24841.png', '24854.png', '22245.png', '17399.png', '08269.png', '21439.png', '24736.png', '12469.png', '18533.png', '09070.png', '19449.png', '21697.png', '10671.png', '02461.png', '14723.png', '18059.png', '03544.png', '15646.png', '18215.png', '10951.png', '15605.png', '22753.png', '03937.png', '18861.png', '03995.png', '03207.png', '24382.png', '06798.png', '16168.png', '15423.png', '09430.png', '13656.png', '23752.png', '02699.png', '00134.png', '01249.png', '08136.png', '08336.png', '09043.png', '22856.png', '15120.png', '16316.png', '02802.png', '08353.png', '18983.png', '21851.png', '21203.png', '13072.png', '15895.png', '07985.png', '14000.png', '01885.png', '21649.png', '12525.png', '13135.png', '13501.png', '01807.png', '15875.png', '20656.png', '22947.png', '15886.png', '19315.png', '11399.png', '02018.png', '00322.png', '11902.png', '14995.png', '08475.png', '19632.png', '14476.png', '03367.png', '03287.png', '14795.png', '18199.png', '04974.png', '04636.png', '14303.png', '20907.png', '17029.png', '24830.png', '18595.png', '10173.png', '02266.png', '12253.png', '05012.png', '21071.png', '24390.png', '01322.png', '05673.png', '15452.png', '18411.png', '07265.png', '04928.png', '13309.png', '02314.png', '22631.png', '05155.png', '08669.png', '03176.png', '09763.png', '10430.png', '12736.png', '15130.png', '14666.png', '06529.png', '16754.png', '09547.png', '19266.png', '22160.png', '12835.png', '24082.png', '22847.png', '21445.png', '20882.png', '08576.png', '16572.png', '08942.png', '05334.png', '21060.png', '14699.png', '20710.png', '07396.png', '00396.png', '11498.png', '05839.png', '04241.png', '23579.png', '03749.png', '07325.png', '03438.png', '20203.png', '20177.png', '02047.png', '06554.png', '15852.png', '12505.png', '08399.png', '03128.png', '01567.png', '13581.png', '02505.png', '16518.png', '15804.png', '20780.png', '11642.png', '24826.png', '23415.png', '01859.png', '15031.png', '17053.png', '23831.png', '20391.png', '07373.png', '19032.png', '06773.png', '19249.png', '10131.png', '00964.png', '17255.png', '04263.png', '13314.png', '11496.png', '10567.png', '07759.png', '00373.png', '03431.png', '01763.png', '08910.png', '11663.png', '05901.png', '16676.png', '13235.png', '18155.png', '08109.png', '00552.png', '00755.png', '05786.png', '23137.png', '09138.png', '22804.png', '14086.png', '05463.png', '11012.png', '06370.png', '09812.png', '13817.png', '17414.png', '16828.png', '19845.png', '21542.png', '19995.png', '20176.png', '14969.png', '10516.png', '22490.png', '03747.png', '18978.png', '21612.png', '14034.png', '16285.png', '11687.png', '07699.png', '18134.png', '12382.png', '21083.png', '17428.png', '23352.png', '10282.png', '08505.png', '09574.png', '15324.png', '19815.png', '07258.png', '20188.png', '16108.png', '02300.png', '10079.png', '15756.png', '00419.png', '04791.png', '11056.png', '00385.png', '12314.png', '13421.png', '07533.png', '16905.png', '20099.png', '04324.png', '13964.png', '04837.png', '06797.png', '04087.png', '17720.png', '04710.png', '14678.png', '01743.png', '22969.png', '23821.png', '03211.png', '23107.png', '05569.png', '20434.png', '18169.png', '18550.png', '06534.png', '11804.png', '07364.png', '13406.png', '06056.png', '12891.png', '02737.png', '23825.png', '12659.png', '10035.png', '07987.png', '10821.png', '22610.png', '00066.png', '08121.png', '16330.png', '16150.png', '17351.png', '11654.png', '10859.png', '04014.png', '15401.png', '05241.png', '05005.png', '04728.png', '18613.png', '11524.png', '21162.png', '16187.png', '22727.png', '19860.png', '05927.png', '06677.png', '04536.png', '10084.png', '21770.png', '19795.png', '06955.png', '19701.png', '22944.png', '16760.png', '04581.png', '24522.png', '06600.png', '02292.png', '22090.png', '13046.png', '15111.png', '06460.png', '15302.png', '19335.png', '01437.png', '06530.png', '02779.png', '00234.png', '09514.png', '09606.png', '23403.png', '04232.png', '14897.png', '06445.png', '06679.png', '03504.png', '24265.png', '11730.png', '24805.png', '02245.png', '19406.png', '01116.png', '09517.png', '17733.png', '00994.png', '23894.png', '22216.png', '17709.png', '07679.png', '14575.png', '02136.png', '18444.png', '21474.png', '12339.png', '21276.png', '15547.png', '18099.png', '00101.png', '04629.png', '06521.png', '08056.png', '17444.png', '02074.png', '11785.png', '03223.png', '20404.png', '20062.png', '09701.png', '07896.png', '00003.png', '10576.png', '02809.png', '20437.png', '05726.png', '23174.png', '22217.png', '00770.png', '06041.png', '21299.png', '10177.png', '10201.png', '10257.png', '04344.png', '01821.png', '12727.png', '04999.png', '01634.png', '12864.png', '05972.png', '22269.png', '01288.png', '02683.png', '04891.png', '21585.png', '03070.png', '16044.png', '10405.png', '02635.png', '18328.png', '06346.png', '21637.png', '01662.png', '14430.png', '02562.png', '16308.png', '08891.png', '02298.png', '13114.png', '19399.png', '20100.png', '21679.png', '17538.png', '19118.png', '07676.png', '04363.png', '01408.png', '04326.png', '00171.png', '06947.png', '04332.png', '19529.png', '14069.png', '23016.png', '23759.png', '08823.png', '23880.png', '13431.png', '16392.png', '12123.png', '22239.png', '00454.png', '06144.png', '01762.png', '14636.png', '23298.png', '13233.png', '08931.png', '16656.png', '20775.png', '00306.png', '19979.png', '23250.png', '06755.png', '08240.png', '24273.png', '11859.png', '13511.png', '14649.png', '16096.png', '00887.png', '03019.png', '20801.png', '07427.png', '09881.png', '16035.png', '10754.png', '07383.png', '08645.png', '18384.png', '05321.png', '13545.png', '11803.png', '00798.png', '07624.png', '09934.png', '17869.png', '21633.png', '21357.png', '00911.png', '20520.png', '10692.png', '07848.png', '17505.png', '11263.png', '01606.png', '12443.png', '15910.png', '07464.png', '21498.png', '12983.png', '24335.png', '12346.png', '16783.png', '09257.png', '18591.png', '11059.png', '05568.png', '00594.png', '15512.png', '03573.png', '20998.png', '02041.png', '01646.png', '17361.png', '22866.png', '22180.png', '21929.png', '21702.png', '03150.png', '10764.png', '22110.png', '22360.png', '05900.png', '21483.png', '11072.png', '23094.png', '22912.png', '20018.png', '23887.png', '22665.png', '02726.png', '21074.png', '04338.png', '15677.png', '24286.png', '04150.png', '10396.png', '05312.png', '22791.png', '08338.png', '13381.png', '01758.png', '03949.png', '07626.png', '00450.png', '04296.png', '21719.png', '14865.png', '21479.png', '13742.png', '11088.png', '09294.png', '15757.png', '22525.png', '20789.png', '02829.png', '01415.png', '03427.png', '16401.png', '05524.png', '12749.png', '23237.png', '21387.png', '07196.png', '02912.png', '00166.png', '04190.png', '04798.png', '08390.png', '08013.png', '11410.png', '04590.png', '23679.png', '02597.png', '08889.png', '00517.png', '23754.png', '12957.png', '21523.png', '02360.png', '05237.png', '14127.png', '09473.png', '03178.png', '09193.png', '00569.png', '11779.png', '09343.png', '08473.png', '24788.png', '12547.png', '20657.png', '16236.png', '07023.png', '07299.png', '12216.png', '01892.png', '01848.png', '07323.png', '18017.png', '08739.png', '21241.png', '13995.png', '10278.png', '11572.png', '16194.png', '08326.png', '18104.png', '18348.png', '15660.png', '07877.png', '21914.png', '20345.png', '02303.png', '10455.png', '02565.png', '22044.png', '04072.png', '00516.png', '20688.png', '04596.png', '14359.png', '19881.png', '00448.png', '20070.png', '24503.png', '04556.png', '20123.png', '22799.png', '23553.png', '10844.png', '06710.png', '13871.png', '08999.png', '01363.png', '23392.png', '23370.png', '22938.png', '21689.png', '05533.png', '14211.png', '05153.png', '17308.png', '23127.png', '01780.png', '21938.png', '22088.png', '18664.png', '07066.png', '10545.png', '21790.png', '16026.png', '10281.png', '00688.png', '13830.png', '21640.png', '05090.png', '19323.png', '15283.png', '04424.png', '16536.png', '11130.png', '21113.png', '16880.png', '00411.png', '15448.png', '04139.png', '00136.png', '05184.png', '04800.png', '21846.png', '10390.png', '23688.png', '23211.png', '23799.png', '19277.png', '22870.png', '14947.png', '09037.png', '07777.png', '23583.png', '12708.png', '15176.png', '00497.png', '08391.png', '04412.png', '24022.png', '14738.png', '06070.png', '21064.png', '13904.png', '03786.png', '02368.png', '21117.png', '04866.png', '15614.png', '04954.png', '01386.png', '19856.png', '17853.png', '23527.png', '11186.png', '23141.png', '12919.png', '21499.png', '11786.png', '08488.png', '17391.png', '23079.png', '16700.png', '10646.png', '19604.png', '09165.png', '20106.png', '20883.png', '17227.png', '14247.png', '05582.png', '09622.png', '24896.png', '07240.png', '06480.png', '24652.png', '09618.png', '10675.png', '17266.png', '23157.png', '14283.png', '10295.png', '16769.png', '07933.png', '05867.png', '07584.png', '17899.png', '03787.png', '16963.png', '21896.png', '18771.png', '14494.png', '13721.png', '09399.png', '23518.png', '18730.png', '10947.png', '12245.png', '21251.png', '20143.png', '20012.png', '17015.png', '15292.png', '06409.png', '08550.png', '18902.png', '22431.png', '15617.png', '23665.png', '04013.png', '13850.png', '09414.png', '19948.png', '05844.png', '11560.png', '12435.png', '23708.png', '05603.png', '09475.png', '20644.png', '11745.png', '06266.png', '13207.png', '12310.png', '13445.png', '23604.png', '09518.png', '15041.png', '07336.png', '08689.png', '23185.png', '20845.png', '19290.png', '22858.png', '15357.png', '17684.png', '06577.png', '02083.png', '13263.png', '24280.png', '15032.png', '03130.png', '12492.png', '15455.png', '06133.png', '24923.png', '05189.png', '09532.png', '22733.png', '06893.png', '03511.png', '22655.png', '12867.png', '15521.png', '14819.png', '13143.png', '11004.png', '09466.png', '17433.png', '22235.png', '03551.png', '12801.png', '18768.png', '22353.png', '09529.png', '18599.png', '06181.png', '19397.png', '17994.png', '14876.png', '03423.png', '16545.png', '05581.png', '05578.png', '24853.png', '15749.png', '23807.png', '07217.png', '05052.png', '18622.png', '08143.png', '20110.png', '24656.png', '00502.png', '21664.png', '20173.png', '13116.png', '18760.png', '18669.png', '13712.png', '02178.png', '11177.png', '20207.png', '14773.png', '22746.png', '15989.png', '19902.png', '01298.png', '20747.png', '19445.png', '09017.png', '14191.png', '07980.png', '07716.png', '02455.png', '05118.png', '04077.png', '01237.png', '22151.png', '10938.png', '19459.png', '09149.png', '08625.png', '10637.png', '08060.png', '11982.png', '17237.png', '16057.png', '13051.png', '05120.png', '03713.png', '04990.png', '13901.png', '07006.png', '00718.png', '04252.png', '15236.png', '11216.png', '17165.png', '11229.png', '23801.png', '21136.png', '16890.png', '16097.png', '15922.png', '20540.png', '15152.png', '11776.png', '17358.png', '20774.png', '01586.png', '10960.png', '09161.png', '11578.png', '24367.png', '03920.png', '21036.png', '14013.png', '21383.png', '18667.png', '20416.png', '23057.png', '23186.png', '05571.png', '16580.png', '15310.png', '22549.png', '00812.png', '00342.png', '01175.png', '11714.png', '12834.png', '07386.png', '11607.png', '17934.png', '06400.png', '17435.png', '19672.png', '13573.png', '16674.png', '07390.png', '15336.png', '11066.png', '15714.png', '03114.png', '05847.png', '15768.png', '17596.png', '24415.png', '10276.png', '10400.png', '20144.png', '06814.png', '07757.png', '04037.png', '08266.png', '17383.png', '09804.png', '18166.png', '05266.png', '10992.png', '16149.png', '18777.png', '19489.png', '01566.png', '17560.png', '02450.png', '02487.png', '24729.png', '10100.png', '23569.png', '07406.png', '24267.png', '16579.png', '24007.png', '11529.png', '05780.png', '18761.png', '00859.png', '22429.png', '00726.png', '15039.png', '01727.png', '17246.png', '15351.png', '02804.png', '15988.png', '17657.png', '07369.png', '07754.png', '09412.png', '01242.png', '13520.png', '21728.png', '05813.png', '10369.png', '05383.png', '23624.png', '01950.png', '18165.png', '08585.png', '12602.png', '02286.png', '21412.png', '05731.png', '14033.png', '11146.png', '24132.png', '15475.png', '14546.png', '16776.png', '07959.png', '10323.png', '12725.png', '09242.png', '07546.png', '00393.png', '00606.png', '10261.png', '02270.png', '08895.png', '04712.png', '23565.png', '00334.png', '17379.png', '10815.png', '22755.png', '00263.png', '01867.png', '13208.png', '17122.png', '06441.png', '23691.png', '01481.png', '15079.png', '15668.png', '06423.png', '24019.png', '16051.png', '20543.png', '03234.png', '13478.png', '19121.png', '06867.png', '13042.png', '11918.png', '13355.png', '20213.png', '20625.png', '23365.png', '12301.png', '16389.png', '21906.png', '19186.png', '18835.png', '02780.png', '10846.png', '06771.png', '02310.png', '13330.png', '16317.png', '14811.png', '11955.png', '24033.png', '00705.png', '03959.png', '18593.png', '04289.png', '11840.png', '14492.png', '02730.png', '00346.png', '24285.png', '14184.png', '08993.png', '04820.png', '24769.png', '13405.png', '14041.png', '10148.png', '02405.png', '12489.png', '05215.png', '12133.png', '19433.png', '22406.png', '23385.png', '23703.png', '02741.png', '05065.png', '16371.png', '05263.png', '19667.png', '04244.png', '08132.png', '19141.png', '10579.png', '01583.png', '07783.png', '12816.png', '15356.png', '19767.png', '03744.png', '12099.png', '16788.png', '10352.png', '10161.png', '07600.png', '00372.png', '18299.png', '19051.png', '04029.png', '00818.png', '14909.png', '08368.png', '04448.png', '19080.png', '22628.png', '06836.png', '13827.png', '01981.png', '18252.png', '22760.png', '00116.png', '10289.png', '02898.png', '02159.png', '24917.png', '18387.png', '05188.png', '14319.png', '22775.png', '07129.png', '08226.png', '02391.png', '01236.png', '24332.png', '06498.png', '13873.png', '23657.png', '11570.png', '12630.png', '12971.png', '10757.png', '06047.png', '18154.png', '21016.png', '00308.png', '23059.png', '10050.png', '15044.png', '17939.png', '19062.png', '14163.png', '22536.png', '03168.png', '07553.png', '15393.png', '22143.png', '03468.png', '08009.png', '13677.png', '01957.png', '19600.png', '16588.png', '10152.png', '06937.png', '15545.png', '00832.png', '10565.png', '15139.png', '11713.png', '14605.png', '13713.png', '05008.png', '08169.png', '07511.png', '01025.png', '12334.png', '17094.png', '06065.png', '20660.png', '07026.png', '01397.png', '10028.png', '24838.png', '06028.png', '05459.png', '23811.png', '06187.png', '07035.png', '18316.png', '05790.png', '18779.png', '12053.png', '06119.png', '22226.png', '07010.png', '13282.png', '04291.png', '18496.png', '02088.png', '17044.png', '14569.png', '11199.png', '01901.png', '17656.png', '03115.png', '22657.png', '23294.png', '13299.png', '01173.png', '16448.png', '14965.png', '11023.png', '13168.png', '18402.png', '05612.png', '00383.png', '06011.png', '22467.png', '09501.png', '05701.png', '02757.png', '03226.png', '12307.png', '02108.png', '24177.png', '22825.png', '07441.png', '11670.png', '06992.png', '10107.png', '19195.png', '09805.png', '03919.png', '03448.png', '24722.png', '05604.png', '12907.png', '10220.png', '22750.png', '15553.png', '15502.png', '11566.png', '24593.png', '08138.png', '13773.png', '04016.png', '12818.png', '06617.png', '15957.png', '20888.png', '03552.png', '22066.png', '01342.png', '05057.png', '10632.png', '03698.png', '13660.png', '13064.png', '02995.png', '06873.png', '13693.png', '11453.png', '10040.png', '16348.png', '14928.png', '05051.png', '18468.png', '02694.png', '02973.png', '08310.png', '24334.png', '08041.png', '12423.png', '21752.png', '02707.png', '13424.png', '19570.png', '15681.png', '08025.png', '20448.png', '07107.png', '21503.png', '20482.png', '21519.png', '02101.png', '23123.png', '03301.png', '04909.png', '17582.png', '15848.png', '22198.png', '00130.png', '13106.png', '15955.png', '01029.png', '01337.png', '06228.png', '19538.png', '05640.png', '20382.png', '20595.png', '01613.png', '21514.png', '07894.png', '21138.png', '09857.png', '08756.png', '20592.png', '15992.png', '19786.png', '10377.png', '13893.png', '19987.png', '24235.png', '11201.png', '07502.png', '11946.png', '14508.png', '19212.png', '24433.png', '09722.png', '17479.png', '15678.png', '07891.png', '13103.png', '10574.png', '02923.png', '00188.png', '18876.png', '14398.png', '21859.png', '02003.png', '07241.png', '05542.png', '21815.png', '15997.png', '24098.png', '02622.png', '01795.png', '19993.png', '06788.png', '14418.png', '22680.png', '17551.png', '22425.png', '02470.png', '05073.png', '24824.png', '16496.png', '14715.png', '03029.png', '05041.png', '21525.png', '09844.png', '19409.png', '16703.png', '15639.png', '01219.png', '11970.png', '06951.png', '05029.png', '22851.png', '13837.png', '21252.png', '12849.png', '10702.png', '11531.png', '02568.png', '02205.png', '12209.png', '19847.png', '11896.png', '10056.png', '06432.png', '18187.png', '10343.png', '20916.png', '19084.png', '03028.png', '04647.png', '07539.png', '13362.png', '10269.png', '16913.png', '07808.png', '21103.png', '16127.png', '04403.png', '01399.png', '13120.png', '16780.png', '04732.png', '18594.png', '16142.png', '23918.png', '14906.png', '15641.png', '04840.png', '20094.png', '16438.png', '07935.png', '12959.png', '20352.png', '09549.png', '16910.png', '07080.png', '12790.png', '03443.png', '07453.png', '14720.png', '19413.png', '16530.png', '14943.png', '02868.png', '05076.png', '09524.png', '15765.png', '04054.png', '09256.png', '11640.png', '22621.png', '23340.png', '09417.png', '19036.png', '10956.png', '16109.png', '00043.png', '19221.png', '08414.png', '22037.png', '05768.png', '08176.png', '08578.png', '22352.png', '21639.png', '11236.png', '02375.png', '20291.png', '12504.png', '10071.png', '23897.png', '02415.png', '18042.png', '24133.png', '06888.png', '06817.png', '16883.png', '11137.png', '06394.png', '07523.png', '00634.png', '24714.png', '09946.png', '14967.png', '00318.png', '18207.png', '07827.png', '02369.png', '11361.png', '16931.png', '11462.png', '03233.png', '15752.png', '08284.png', '24868.png', '09795.png', '08348.png', '04298.png', '07703.png', '16307.png', '20973.png', '02740.png', '03783.png', '07756.png', '19930.png', '09919.png', '14405.png', '07544.png', '21424.png', '10746.png', '01040.png', '20420.png', '04968.png', '22571.png', '16736.png', '03386.png', '09891.png', '00227.png', '22102.png', '06506.png', '02603.png', '24496.png', '10435.png', '18416.png', '02321.png', '00908.png', '11939.png', '11844.png', '09456.png', '19921.png', '09782.png', '17449.png', '13149.png', '17315.png', '09686.png', '18499.png', '21340.png', '16233.png', '04273.png', '10674.png', '24110.png', '13877.png', '06591.png', '05103.png', '01951.png', '07957.png', '08271.png', '12056.png', '00769.png', '00944.png', '02760.png', '23541.png', '10036.png', '09014.png', '19897.png', '14758.png', '09918.png', '14371.png', '05620.png', '10367.png', '21062.png', '05511.png', '06024.png', '01678.png', '22619.png', '20021.png', '23738.png', '15157.png', '22781.png', '06499.png', '24143.png', '09683.png', '01489.png', '07878.png', '22470.png', '12352.png', '05167.png', '07423.png', '12657.png', '05615.png', '01700.png', '01559.png', '19152.png', '17648.png', '02174.png', '10557.png', '02768.png', '13050.png', '07978.png', '10088.png', '05657.png', '06898.png', '14403.png', '12162.png', '00577.png', '12769.png', '22351.png', '24660.png', '17757.png', '16254.png', '06253.png', '20878.png', '19516.png', '14645.png', '05994.png', '03488.png', '07833.png', '12092.png', '09814.png', '12595.png', '20370.png', '05478.png', '18047.png', '06915.png', '14378.png', '14484.png', '16923.png', '15314.png', '04978.png', '13022.png', '00986.png', '13142.png', '01178.png', '18566.png', '18453.png', '16515.png', '21159.png', '04705.png', '10469.png', '19939.png', '21310.png', '05953.png', '09243.png', '21823.png', '20702.png', '10918.png', '24179.png', '24210.png', '06447.png', '09550.png', '01817.png', '15102.png', '18037.png', '10450.png', '01304.png', '10195.png', '05513.png', '07711.png', '09292.png', '24060.png', '07658.png', '17074.png', '20683.png', '01799.png', '09902.png', '13237.png', '24888.png', '01055.png', '11420.png', '23044.png', '00971.png', '14971.png', '03198.png', '07267.png', '01719.png', '07069.png', '23740.png', '21041.png', '20107.png', '19840.png', '19973.png', '20611.png', '02230.png', '23957.png', '18119.png', '15016.png', '06442.png', '19549.png', '16155.png', '08260.png', '05409.png', '14299.png', '22865.png', '04741.png', '18924.png', '09621.png', '05194.png', '23704.png', '07807.png', '12789.png', '18336.png', '00417.png', '18587.png', '14084.png', '00855.png', '00766.png', '14890.png', '16460.png', '12909.png', '07564.png', '02260.png', '03212.png', '16031.png', '00292.png', '22498.png', '07430.png', '00174.png', '23662.png', '23494.png', '20362.png', '13921.png', '01961.png', '02380.png', '17889.png', '23720.png', '06162.png', '02966.png', '12591.png', '24321.png', '08170.png', '15974.png', '13744.png', '22554.png', '12787.png', '18848.png', '14566.png', '02657.png', '00710.png', '01218.png', '21535.png', '06212.png', '17629.png', '15353.png', '15195.png', '24431.png', '00533.png', '18268.png', '06126.png', '18956.png', '11044.png', '02107.png', '06410.png', '09730.png', '19103.png', '10902.png', '14321.png', '08190.png', '08357.png', '22395.png', '10623.png', '02686.png', '02823.png', '19091.png', '20397.png', '16819.png', '22547.png', '18614.png', '13551.png', '00022.png', '20681.png', '22337.png', '10519.png', '16638.png', '02817.png', '22199.png', '10779.png', '20646.png', '12573.png', '01944.png', '10681.png', '05692.png', '11483.png', '18048.png', '18063.png', '20723.png', '18174.png', '12706.png', '02358.png', '02387.png', '10962.png', '21863.png', '18743.png', '03469.png', '19519.png', '00074.png', '23216.png', '02803.png', '03421.png', '22921.png', '22031.png', '08894.png', '21247.png', '04410.png', '08691.png', '01721.png', '13240.png', '10074.png', '16402.png', '03536.png', '01206.png', '11120.png', '22779.png', '23417.png', '12105.png', '12969.png', '12920.png', '03853.png', '23104.png', '09377.png', '23420.png', '19773.png', '02685.png', '08604.png', '24482.png', '09592.png', '21880.png', '12117.png', '07160.png', '22024.png', '03639.png', '00791.png', '12426.png', '09271.png', '00070.png', '13555.png', '24905.png', '03072.png', '11055.png', '00433.png', '11903.png', '15682.png', '23985.png', '18930.png', '06455.png', '02659.png', '00368.png', '02095.png', '24233.png', '05239.png', '13944.png', '10264.png', '15428.png', '04360.png', '00014.png', '13399.png', '09831.png', '12699.png', '14522.png', '09139.png', '06387.png', '21951.png', '14366.png', '14489.png', '15754.png', '13553.png', '00494.png', '05081.png', '00862.png', '16542.png', '12386.png', '07593.png', '01454.png', '16948.png', '24370.png', '01124.png', '13401.png', '19024.png', '01136.png', '20913.png', '03378.png', '23961.png', '23027.png', '06818.png', '12398.png', '12773.png', '16242.png', '16137.png', '02082.png', '11172.png', '07905.png', '24199.png', '23205.png', '12998.png', '21683.png', '04997.png', '20976.png', '02756.png', '15049.png', '24653.png', '20979.png', '09698.png', '17549.png', '07765.png', '12393.png', '19573.png', '24629.png', '15359.png', '24477.png', '10665.png', '21674.png', '23052.png', '24131.png', '23355.png', '03295.png', '05548.png', '05976.png', '09963.png', '23816.png', '15415.png', '16729.png', '21563.png', '16487.png', '09653.png', '13279.png', '23387.png', '21463.png', '10920.png', '17527.png', '12181.png', '17837.png', '21870.png', '08370.png', '22966.png', '06777.png', '24237.png', '19443.png', '16598.png', '18106.png', '12385.png', '05224.png', '03518.png', '09989.png', '23086.png', '22477.png', '10373.png', '15395.png', '08098.png', '03868.png', '03657.png', '19124.png', '19038.png', '08580.png', '17331.png', '07606.png', '14068.png', '04094.png', '02633.png', '17278.png', '15108.png', '16906.png', '03896.png', '22789.png', '17777.png', '02994.png', '14091.png', '07817.png', '22933.png', '23318.png', '18143.png', '17932.png', '20945.png', '04713.png', '13897.png', '02827.png', '09044.png', '14142.png', '06275.png', '19907.png', '04771.png', '04875.png', '03035.png', '03238.png', '09528.png']
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
            logits_u_w_65 = (ema_model(inputs_u_w)[0])
            logits_u_w = interp(logits_u_w_65)
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())
            logits_u_w_65, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w_65.detach()) 
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)
            pseudo_label_65 = torch.softmax(logits_u_w_65.detach(), dim=1)
            max_probs_65, targets_u_w_65 = torch.max(pseudo_label_65, dim=1)

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
        pixelWiseWeight[pixelWiseWeight != 1.0] == 0.0
        src_pixels = F.interpolate(pixelWiseWeight.unsqueeze(1),
                                            size=(65,65), mode='nearest').squeeze(1)

        _, _, Ht, Wt = tgt_feat.size()
        tgt_mask = F.interpolate(targets_u.unsqueeze(1).float(), size=(65,65), mode='nearest').squeeze(1).long()
        tgt_mask_upt = copy.deepcopy(tgt_mask)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_mask_upt[(((max_probs_65 < cfg.SOLVER.DELTA) * (targets_u_w_65 == i)).int() + (src_pixels != 1.0).int()) == 2] = 255
        if i_iter >= 100 and i_iter <= 300 and i_iter %20 == 0:
            print((src_pixels != 1.0).sum())
            print((tgt_mask_upt == 255).sum())
            print(((max_probs_65 < cfg.SOLVER.DELTA) * (targets_u_w_65 == i)).sum())
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
