import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from scipy import interp
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict


class AdeDataSet(Dataset):
    def __init__(self, csv_file):
        self.len = len(csv_file)
        self.wsi_feats = csv_file.iloc[:, 0]
        self.wsi_labels = csv_file.iloc[:, 1]

    def __getitem__(self, index):
        single_wsi_feats_path = self.wsi_feats[index]
        single_wsi_feats = pd.read_csv(single_wsi_feats_path)
        single_wsi_feats = shuffle(single_wsi_feats).reset_index(drop=True)
        feats = np.float32(single_wsi_feats.to_numpy())

        single_wsi_label = self.wsi_labels[index]
        label = np.float32(np.zeros(6))
        if int(single_wsi_label) <= (len(label) - 1):
            label[int(single_wsi_label)] = 1

        return feats, label

    def __len__(self):
        return self.len



def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)

    # 2.Calculating weight
    num_cls = args.num_classes
    weight_dict = {}
    for i in range(num_cls):
        weight_dict[i] = csvs.shape[0] / (csvs.loc[csvs['label'] == i].shape[0])
    # 3.Add weight column
    csvs['weight'] = csvs['label'].apply(lambda x: weight_dict[x])
    # 4.Define the WeightedRandomSampler sampler
    data_weight = torch.tensor(csvs['weight'].to_numpy(), dtype=torch.float)
    num_samples = csvs.shape[0]  # How many samples were taken? It can be set to the same amount as the dataset or set by yourself.
    # Define the sampler, pass in the prepared weight array, sample total, and select Put back sampling
    sampler = torch.utils.data.sampler.WeightedRandomSampler(data_weight, num_samples, replacement=True)
    # 5.Set up Dataset and Dataloader
    dataset = AdeDataSet(csvs)
    train_data_iter = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=1)

    total_loss = 0
    bc = 0
    for i, data in enumerate(train_data_iter, 0):
        optimizer.zero_grad()
        bag_feats, bag_label = data
        bag_feats = bag_feats.cuda()
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))

        loss = 0.5 * max_loss + 0.5 * bag_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)



def test(epoch, test_df, milnet, criterion, optimizer, args):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []

    dataset = AdeDataSet(csvs)
    test_data_iter = DataLoader(dataset=dataset, batch_size=1)
    with torch.no_grad():
        for i, data in enumerate(test_data_iter, 0):
            bag_feats, bag_label = data
            bag_feats = bag_feats.cuda()
            bag_label = bag_label.cuda()
            label = bag_label.squeeze().cpu().numpy()
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5 * max_loss + 0.5 * bag_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            test_predictions.extend(
                [(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # compute accuracy and F1score
    F1score = dict()
    label_array = [np.argmax(test_label) for test_label in test_labels]
    prediction_array = [np.argmax(test_prediction) for test_prediction in test_predictions]

    acc_value = accuracy_score(label_array, prediction_array)




    F1score["macro"] = f1_score(label_array, prediction_array, average='macro')
    F1score["micro"] = f1_score(label_array, prediction_array, average='micro')

    auc_value, _, thresholds_optimal = multi_label_roc(epoch, test_labels, test_predictions, args.num_classes)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)

    return acc_value, F1score, total_loss / len(test_df), avg_score, auc_value, thresholds_optimal





def multi_label_roc(epoch, labels, predictions, num_classes, pos_label=1):
    # fprs = []
    # tprs = []

    fprs = dict()
    tprs = dict()
    roc_auc = dict()

    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        # fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr, tpr, threshold = roc_curve(label, prediction)

        fprs[c], tprs[c] = fpr, tpr
        roc_auc[c] = auc(fprs[c], tprs[c])

        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)

    # --------------------------------------------------------------------------------------------------------------#
    # Compute micro-average ROC curve and ROC area（方法二）
    fprs["micro"], tprs["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fprs["micro"], tprs["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fprs[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fprs[i], tprs[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fprs["macro"] = all_fpr
    tprs["macro"] = mean_tpr
    roc_auc["macro"] = auc(fprs["macro"], tprs["macro"])


    return aucs, thresholds, thresholds_optimal



def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=6, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset_train', default='Ade', type=str, help='Dataset folder name')
    parser.add_argument('--dataset_val', default='Ade_val', type=str, help='Dataset folder name')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                   dropout_v=args.dropout_node).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    bags_csv_train = os.path.join('datasets', args.dataset_train, args.dataset_train + '.csv')
    bags_csv_val = os.path.join('datasets', args.dataset_val, args.dataset_val + '.csv')

    train_path = pd.read_csv(bags_csv_train)
    test_path = pd.read_csv(bags_csv_val)

    best_accuracy = 0
    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))
    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args)  # iterate all bags
        accuracy, F1score, test_loss_bag, avg_score, aucs, thresholds_optimal = test(epoch, test_path, milnet,
                                                                                     criterion, optimizer, args)

        print(
            '\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, F1score_macro: %.4f, F1score_micro: %.4f, average score: %.4f, AUC: ' %
            (epoch, args.num_epochs, train_loss_bag, test_loss_bag, accuracy, F1score["macro"], F1score["micro"],
             avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))

        scheduler.step()
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            save_name = os.path.join(save_path, str(run + 1) + '.pth')
            torch.save(milnet.state_dict(), save_name)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> ' + '|'.join(
                'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))


if __name__ == '__main__':
    main()