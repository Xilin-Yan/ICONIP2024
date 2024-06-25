import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time
from archs import UNext
import torch
import random
import numpy as np

# 设置随机种子
seed = 77

np.random.seed(seed)  # 设置 NumPy 中的随机种子
random.seed(seed)  # 设置 Python 标准库中的随机种子，以确保其他 Python 函数中使用的随机数也是可复现的。

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 设置 PyTorch 在 CUDA 环境下的随机种子，以确保 CUDA 计算的结果是可复现的。
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，此命令将确保所有的 GPU 使用相同的随机种子。
torch.backends.cudnn.deterministic = True  # 确保在使用 cuDNN 加速时结果可复现，但可能会降低性能。
torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动寻找最适合当前配置的高效算法的功能，以确保结果的一致性。


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model = torch.load('models/%s/model.pth' % args.name)

    model = model.cuda()
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', args.name, str(c)), exist_ok=True)

    dices = []
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            dices.append(dice)

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', args.name, str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    output_flag = False

    if output_flag:
        dir_name = './models/%s' % args.name.split('_sample')[0]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        sample_ratio = args.name.split('_sample')[1]

        with open(os.path.join(dir_name, '%s.txt' % str(sample_ratio)), 'w') as f:
            for i in range(len(dices)):
                f.write(str(dices[i]) + '\n')

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
