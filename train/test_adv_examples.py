from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import utils
import numpy as np
import torch
import torch.multiprocessing as mp
# from hparams import hp
# from resnet import resnet50
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm
import torchvision.models as models
from tqdm import tqdm
import transforms
from PIL import Image
# from ayang_net import AyangNet

from data_adv import CIFAR
from inception import inception_v3
from model_simple import SimpleConvNet
# from inception2 import inception_v4
# from wideresnet import WideResNet
# from densenet import DenseNet
import time

def reconstruct_image(im):
    im = im.numpy()
    im = np.transpose(im, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = 256 * (im * std + mean)
    return im


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.3 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


NAME_TO_MODEL = {
    # 'resnet50': resnet50(num_classes=100),
    # 'ayangnet': AyangNet(),
    # 'densenet': models.densenet161(num_classes=100),
    'inceptionv3': inception_v3(num_classes=10),
    'simplenet': SimpleConvNet(num_classes=10)
    # 'inceptionv4': inception_v4(num_classes=100),
    # 'wideresnet': WideResNet(28, 100, widen_factor=10),
    # 'widedensenet': DenseNet(60, (6, 6, 6, 6), 64, num_classes=100)
}


if __name__ == '__main__':
    default_path = '../data'
    noise_decay = 0.55
    loss_fn = CrossEntropyLoss()

    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'first')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')

    # dataset
    parser.add_argument('--data_path', default = default_path)
    parser.add_argument('--synset', default = '')
    parser.add_argument('--categories', default = '../data/categories.txt')

    # training
    parser.add_argument('--epochs', default = 500, type = int)
    parser.add_argument('--batch', default = 16, type = int)
    parser.add_argument('--snapshot', default = 2, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--gpu', default = '7')
    parser.add_argument('--name', default = 'simplenet')

    # Training Parameters
    parser.add_argument('--lr', default = 0.001, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--weight_decay', default = 1e-5, type = float)
    parser.add_argument('--noise', default = 0, type = float)

    # parse arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    args.gpu = '0'
    args.name = 'simplenet'
    # args.adv_name = 'simplenet'
    args.resume = './exp/adversary-simple-model-trial-3/best.pth'


    # set up gpus for training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set up datasets and loaders
    data, loaders = {}, {}
    for split in ['val']:
        data[split] = CIFAR(data_path = os.path.join(args.data_path, args.synset), split = split, augment= False, filename='adversarial_examples.npz')
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0}'.format(len(data['val'])))

    # set up map for different categories
    # categories = np.genfromtxt(args.categories, dtype='str')[:, 0]

    # set up model and convert into cuda
    model = NAME_TO_MODEL[args.name].cuda()
    print('==> model loaded')
    best_top_1 = 0


    # set up experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = args.clean)
    logger = utils.Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))

    # load snapshot of model and optimizer
    if args.resume is not None:
        if os.path.isfile(args.resume):
            snapshot = torch.load(args.resume)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            # If this doesn't work, can use optimizer.load_state_dict
            # optimizer.load_state_dict(snapshot['optimizer'])

        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
    else:
        epoch = 0


    # testing the model
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()


    for images, labels in tqdm(loaders['val'], desc = 'epoch %d' % (epoch + 1)):
        # output_adv = model_adv.forward(Variable(images.cuda())).cpu()

        # tmp = images.numpy()
        # print(tmp.shape)
        # images_model = Image.fromarray(images.numpy())
        outputs = model.forward(Variable(images.cuda())).cpu()

        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

    # if top1.avg > best_top_1:
    #     best_top_1 = top5.avg

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} in validation'.format(top1=top1, top5=top5))
    # logger.scalar_summary('Top 1', top1.avg, epoch)
    # logger.scalar_summary('Top 5', top5.avg, epoch)