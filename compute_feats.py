import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from simclr_swin.models.swin_transformer_simclr import SwinTransformerEmbedder
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle



class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)




def swintransformer_compute_feats(args):
    swin_model = SwinTransformerEmbedder()
    num_feats = 768

    for param in swin_model.parameters():
        param.requires_grad = False

    if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights == 'ImageNet':
        weight_path = os.path.join('simclr_swin', 'pretrain_model', 'swin_tiny_patch4_window7_224.pth')
        state_dict_weights = torch.load(weight_path)["model"]
        state_dict_init = swin_model.state_dict()

        for k in list(state_dict_weights.keys()):
            if "head" in k:
                del state_dict_weights[k]
        swin_model.load_state_dict(state_dict_weights, strict=False)
        os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
        torch.save(state_dict_weights,
                   os.path.join('embedder', args.dataset, args.backbone + '-embedder-ImageNet.pth'))

    swin_model.fc = nn.Identity()


    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = mil.IClassifier(swin_model, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = mil.IClassifier(copy.deepcopy(swin_model), num_feats,
                                         output_class=args.num_classes).cuda()  # deepcopy:一旦复制出来了，就是独立的了

        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights == 'ImageNet':

            print('Use ImageNet features.')

        else:
            weight_path = os.path.join('simclr_swin', 'runs', args.weights_high, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            state_dict_init = i_classifier_h.state_dict()
            for i in range(4):
                state_dict_weights.popitem()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, args.backbone + '-m-embedder-high.pth'))

            weight_path = os.path.join('simclr_swin', 'runs', args.weights_low, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            state_dict_init = i_classifier_l.state_dict()
            for i in range(4):
                state_dict_weights.popitem()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, args.backbone + '-m-embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':
        i_classifier = mil.IClassifier(swin_model, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            print('Use ImageNet features.')

        else:
            if args.weights is not None:
                weight_path = os.path.join('simclr_swin', 'runs', args.weights, 'checkpoints', 'model.pth')
            else:
                weight_path = glob.glob('simclr_swin/runs/*/checkpoints/*.pth')[-1]
            state_dict_weights = torch.load(weight_path)
            state_dict_init = i_classifier.state_dict()
            for i in range(4):
                state_dict_weights.popitem()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, args.weights + '-s-embedder.pth'))
            print('Use pretrained features.')

    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high':
        bags_path = os.path.join('WSI', args.dataset, 'pyramid', '*', '*')
    else:
        bags_path = os.path.join('WSI', args.dataset, 'single', '*', '*')
    feats_path = os.path.join('datasets', args.dataset)
    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)

    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path, 'fusion')
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    n_classes = glob.glob(os.path.join('datasets', args.dataset, '*' + os.path.sep))
    n_classes = sorted(n_classes)
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i
        bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2] + '.csv'), index=False)
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('datasets', args.dataset, args.dataset + '.csv'), index=False)


def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if magnification=='single' or magnification=='low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification=='high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
            print()
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
        
def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None, fusion='fusion'):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(0, num_bags): 
            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)

            for idx, low_patch in enumerate(low_patches):
                high_patches = glob.glob(low_patch.replace('.jpeg', os.sep+'*.jpg')) + glob.glob(low_patch.replace('.jpeg', os.sep+'*.jpeg'))
                high_patches = high_patches + glob.glob(low_patch.replace('.jpg', os.sep+'*.jpg')) + glob.glob(low_patch.replace('.jpg', os.sep+'*.jpeg'))
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])
                        if fusion == 'fusion':
                            feats = feats.cpu().numpy()+0.25*feats_list[idx]
                        if fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), 0.25*feats_list[idx]), axis=-1)
                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            print('\n')            

def main():
    parser = argparse.ArgumentParser(description='Compute Ade features from simclr_swin embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='swintransformer', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='batch', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr_swin/runs/*')
    parser.add_argument('--weights_high', default=None, type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr_swin/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr_swin/runs/[FOLDER]`')
    parser.add_argument('--dataset', default='Ade', type=str, help='Dataset folder name [Ade]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    if args.backbone == 'swintransformer':
        swintransformer_compute_feats(args)


if __name__ == '__main__':
    main()