import os, glob
import shutil
import argparse
import random



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Ade', help='Dataset name')
    parser.add_argument('--val_path', type=str, default='WSI/Ade_val', help='The path to the validation set')
    parser.add_argument('--num_classes', default=6, type=int, help='Number of output classes [2]')
    parser.add_argument('--split', default=0.3, type=float, help='Training/Validation split [0.3]')
    args = parser.parse_args()
    # make dir
    if not os.path.isdir(args.val_path):
        os.mkdir(args.val_path)
    val_pyramid_path = os.path.join(args.val_path, 'pyramid')
    if not os.path.isdir(val_pyramid_path):
        os.mkdir(val_pyramid_path)
    for i in range(args.num_classes):
        cls_path = os.path.join(val_pyramid_path, 'CLASS_'+str(i+1))
        if not os.path.isdir(cls_path):
            os.mkdir(cls_path)

    all_patch_path_temp = os.path.join('.', 'WSI', args.dataset, 'pyramid', '*', '*')
    all_patch_path = glob.glob(all_patch_path_temp)
    random.shuffle(all_patch_path)
    # df = pd.DataFrame(all_patch_path)
    all_val_patch = all_patch_path[int(len(all_patch_path) * (1 - args.split)):]
    for src_path in all_val_patch:
        dest_path = os.path.join(val_pyramid_path, src_path.split('/')[4])
        shutil.move(src_path, dest_path)




if __name__ == "__main__":
    main()