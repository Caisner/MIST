import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
from simclr_swin.models.swin_transformer_simclr import SwinTransformerEmbedder

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from grad_cam_utils import GradCAM, show_cam_on_image, center_crop_img
import cv2
import imgviz
from PIL import Image
import math

class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of output classes')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=768)
    parser.add_argument('--embedder_weights', type=str, default='swintransformer-m-embedder-high.pth')
    parser.add_argument('--aggregator_weights', type=str, default='aggregator.pth')
    parser.add_argument('--img_path', type=str, default='./img')
    parser.add_argument('--img_name', type=str, default='VA.jpg')
    parser.add_argument('--target_category', type=int, default=5)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    swin_model = SwinTransformerEmbedder()
    for param in swin_model.parameters():
        param.requires_grad = True
    swin_model.fc = nn.Identity()


    i_classifier = mil.IClassifier(swin_model, args.feats_size, output_class=args.num_classes).cuda()

    if args.embedder_weights != 'ImageNet':
        state_dict_weights = torch.load(args.embedder_weights)
        new_state_dict = OrderedDict()
        for i in range(4):
            state_dict_weights.popitem()
        state_dict_init = i_classifier.state_dict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)


    state_dict_weights = torch.load(args.aggregator_weights)
    state_dict_init_fc = i_classifier.fc.state_dict()
    state_dict_init_fc["weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_init_fc["bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    i_classifier.fc.load_state_dict(state_dict_init_fc, strict=False)

    target_layers = [i_classifier.feature_extractor.norm]  # swin


    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_size = 224
    img_dir = os.path.join(args.img_path, args.img_name)
    assert os.path.exists(img_dir), "file: '{}' dose not exist.".format(img_dir)
    img = Image.open(img_dir).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (img_size, img_size))

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=i_classifier, target_layers=target_layers, use_cuda=True,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))

    grayscale_cam = cam(input_tensor=input_tensor, target_category=args.target_category)


    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    # plt.imshow(visualization)
    cv2.imwrite(os.path.join(args.img_path, args.img_name.split('.')[0]+'_cam.png'), visualization)
    # plt.savefig('plt.png', format='png')


