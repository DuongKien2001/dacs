import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

from umap import UMAP
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d
from utils.helpers import colorize_mask
from torchvision import transforms
import utils.palette as palette

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA evaluation script")
    parser.add_argument("-m","--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()

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

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(data_list, class_num, dataset, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider",
        "car", "truck", "bus",
        "train", "motorcycle", "bicycle"))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    print(M)
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
    return aveJ

def evaluate(model, dataset, ignore_label=250, save_output_images=False, save_dir=None, input_size=(512,1024), model1=None):

    if dataset == 'cityscapes':
        num_classes = 19
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader( data_path, img_size=input_size, img_mean = IMG_MEAN, is_transform=True, split='val')
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        ignore_label = 250

    elif dataset == 'gta':
        num_classes = 19
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        test_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', img_size=(1280,720), mean=IMG_MEAN)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)
        interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)
        ignore_label = 255

    print('Evaluating, found ' + str(len(testloader)) + ' images.')

    data_list = []
    colorize = VOCColorize()
    #colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta", "violet", "lime", "darkred", "springgreen", "darkcyan", "tan"])
    CityScpates_palette = [128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,
                        250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,
                        0,0,70,0,60,100,0,80,100,0,0,230,119,11,32,128,192,0,0,64,128,128,64,128,0,192,
                        128,128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,
                        128,192,192,128,0,0,64,128,0,64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,
                        128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,192,192,0,192,64,128,192,
                        192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,
                        128,192,192,64,64,64,192,64,64,64,192,64,192,192,64,64,64,192,192,64,192,64,192,
                        192,192,192,192,32,0,0,160,0,0,32,128,0,160,128,0,32,0,128,160,0,128,32,128,128,
                        160,128,128,96,0,0,224,0,0,96,128,0,224,128,0,96,0,128,224,0,128,96,128,128,224,
                        128,128,32,64,0,160,64,0,32,192,0,160,192,0,32,64,128,160,64,128,32,192,128,160,
                        192,128,96,64,0,224,64,0,96,192,0,224,192,0,96,64,128,224,64,128,96,192,128,224,
                        192,128,32,0,64,160,0,64,32,128,64,160,128,64,32,0,192,160,0,192,32,128,192,160,
                        128,192,96,0,64,224,0,64,96,128,64,224,128,64,96,0,192,224,0,192,96,128,192,224,
                        128,192,32,64,64,160,64,64,32,192,64,160,192,64,32,64,192,160,64,192,32,192,192,
                        160,192,192,96,64,64,224,64,64,96,192,64,224,192,64,96,64,192,224,64,192,96,192,
                        192,224,192,192,0,32,0,128,32,0,0,160,0,128,160,0,0,32,128,128,32,128,0,160,128,
                        128,160,128,64,32,0,192,32,0,64,160,0,192,160,0,64,32,128,192,32,128,64,160,128,
                        192,160,128,0,96,0,128,96,0,0,224,0,128,224,0,0,96,128,128,96,128,0,224,128,128,
                        224,128,64,96,0,192,96,0,64,224,0,192,224,0,64,96,128,192,96,128,64,224,128,192,
                        224,128,0,32,64,128,32,64,0,160,64,128,160,64,0,32,192,128,32,192,0,160,192,128,
                        160,192,64,32,64,192,32,64,64,160,64,192,160,64,64,32,192,192,32,192,64,160,192,
                        192,160,192,0,96,64,128,96,64,0,224,64,128,224,64,0,96,192,128,96,192,0,224,192,
                        128,224,192,64,96,64,192,96,64,64,224,64,192,224,64,64,96,192,192,96,192,64,224,
                        192,192,224,192,32,32,0,160,32,0,32,160,0,160,160,0,32,32,128,160,32,128,32,160,
                        128,160,160,128,96,32,0,224,32,0,96,160,0,224,160,0,96,32,128,224,32,128,96,160,
                        128,224,160,128,32,96,0,160,96,0,32,224,0,160,224,0,32,96,128,160,96,128,32,224,
                        128,160,224,128,96,96,0,224,96,0,96,224,0,224,224,0,96,96,128,224,96,128,96,224,
                        128,224,224,128,32,32,64,160,32,64,32,160,64,160,160,64,32,32,192,160,32,192,32,
                        160,192,160,160,192,96,32,64,224,32,64,96,160,64,224,160,64,96,32,192,224,32,192,
                        96,160,192,224,160,192,32,96,64,160,96,64,32,224,64,160,224,64,32,96,192,160,96,
                        192,32,224,192,160,224,192,96,96,64,224,96,64,96,224,64,224,224,64,96,96,192,224,
                        96,192,96,224,192,0,0,0]
    """class_names = ["road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle"]"""

    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, _, _ = batch
        size = size[0]
        with torch.no_grad():
            output, feature  = model(Variable(image).cuda())
            output = interp(output)
            
            label_cuda = Variable(label.long()).cuda()
            criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??
            loss = criterion(output, label_cuda)
            total_loss.append(loss.item())

            output = output.cpu().data[0].numpy()
            feature = feature.cpu().data[0].numpy()
            _,a,b = feature.shape
            feature = feature.reshape([2048, a*b])
            feature=feature.transpose(1,0)

            #umap2d = UMAP(init='random', random_state=0)

            #proj_2d = umap2d.fit_transform(feature)
            label1 = F.interpolate(label.unsqueeze(1).float(), size=(a,b), mode='nearest').squeeze(1).long()
            label1 = label1.cpu().data[0].numpy()
            label1 = label1.reshape(-1)
            
            #   plt.scatter(proj[:,0], proj[:,1], color = colors[i])

            #plt.show()
            #plt.savefig('dacs/'+str(index)+'.png')
            #plt.figure().clear()

            if dataset == 'cityscapes':
                gt = np.asarray(label[0].numpy(), dtype=np.int32)
            elif dataset == 'gta':
                gt = np.asarray(label[0].numpy(), dtype=np.int32)

            
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)

            data_list.append([gt.flatten(), output.flatten()])
            
            if index < 500:  
                """
                save_image(image[0].cpu(),index,'_input',palette.CityScpates_palette)
                _, pred_u_s = torch.max(output1, dim=1)
                #_, pred = torch.max(output1_o, dim=1)
                save_image(pred_u_s[0].cpu(),index,'_pred',palette.CityScpates_palette)
                #save_image(pred[0].cpu(),index,'_pred_o',palette.CityScpates_palette)
                save_image(label[0].cpu(), index,'_label',palette.CityScpates_palette)
                """
                ll = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
                lf = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                for i in range(19):
                    m = label1 == ll[i]
                    f = feature[m]
                    if lf[i] == 0:
                        lf[i] = f
                    else:
                        lf[i] = np.append(lf[i], f, axis = 0)

            if index == 499:
                u = np.append(lf[0], lf[1], axis = 0)
                u = np.append(u, lf[2], axis = 0)
                u = np.append(u, lf[3], axis = 0)
                umap2d = UMAP(init='random', random_state=0)

                proj_2d = umap2d.fit_transform(u)
                plt.scatter(proj_2d[0:lf[0].shape[0],0], proj_2d[0:lf[0].shape[0]:,1], color = [CityScpates_palette[0], CityScpates_palette[1], CityScpates_palette[2]])
                v = lf[0].shape[0]
                for i in range(18):
                    plt.scatter(proj_2d[v:v+lf[i+1].shape[0],0], proj_2d[v:v+lf[i+1].shape[0]:,1], color = [CityScpates_palette[i*3+3], CityScpates_palette[i*3+4], CityScpates_palette[i*3+2]])
                    v = v+lf[i+1].shape[0]
                plt.savefig('dacs/'+'a.png')
                plt.figure().clear()
                print('save success')
                     
            
        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

    if save_dir:
        filename = os.path.join(save_dir, 'result.txt')
    else:
        filename = None
    #mIoU = get_iou(data_list, num_classes, dataset, filename)
    #loss = np.mean(total_loss)
    #return mIoU, loss
    return 0,0

def main():
    """Create the model and start the evaluation process."""

    gpu0 = args.gpu

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #model = torch.nn.DataParallel(Res_Deeplab(num_classes=num_classes), device_ids=args.gpu)
    model = Res_Deeplab(num_classes=num_classes)
    #model1 = Res_Deeplab(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    #checkpoint1 = torch.load(args.model_path_o)
    try:
        model.load_state_dict(checkpoint['ema_model'])
        #model1.load_state_dict(checkpoint1['ema_model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['ema_model'])

    model.cuda()
    model.eval()
    #model1.cuda()
    #model1.eval()

    evaluate(model, dataset, ignore_label=ignore_label, save_output_images=args.save_output_images, save_dir=save_dir, input_size=input_size)


if __name__ == '__main__':
    args = get_arguments()

    config = torch.load(args.model_path)['config']

    dataset = config['dataset']
    #dataset = 'cityscapes'
    if dataset == 'cityscapes':
        num_classes = 19
        input_size = (512,1024)
    if dataset == 'gta':
        num_classes = 19
        input_size = (1280,720)

    ignore_label = config['ignore_label']
    save_dir = os.path.join(*args.model_path.split('/')[:-1])

    main()
