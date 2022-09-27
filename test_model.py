import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from scipy import interp
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict


def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1]) <= (len(label) - 1):
            label[int(csv_file_df.iloc[1])] = 1

    return label, feats



def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(epoch, test_df, milnet, args):
    slide_name = []
    for i in range(0, len(test_df)):
        filename = test_df.iloc[i][0]
        slide_name.append(filename)

    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)

            test_labels.extend([label])
            test_predictions.extend(
                [(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # compute accuracy and F1score
    F1score = dict()
    label_array = [np.argmax(test_label) for test_label in test_labels]
    prediction_array = [np.argmax(test_prediction) for test_prediction in test_predictions]

    result_path = 'inference'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    result_dict = {"slide_name": slide_name, "label": label_array, "prediction": prediction_array}
    result_df = DataFrame(result_dict)  # 将字典转换成为数据框
    result_df.to_csv('inference/test_pred.csv', index=False)

    acc_value = accuracy_score(label_array, prediction_array)
    print(acc_value)

    classes = ['HP', 'SSL', 'TSA', 'TA', 'VTA', 'VA']
    label_ndarray = np.array(label_array)
    prediction_ndarray = np.array(prediction_array)
    cm = confusion_matrix(label_ndarray, prediction_ndarray)
    cm_save_name = 'inference/{}_cm.png'.format(epoch)
    title = 'Confusion matrix of adenoma classification'


    disp_confusion_matrix(cm, cm_save_name, classes=classes, title=title)



    F1score["macro"] = f1_score(label_array, prediction_array, average='macro')
    F1score["micro"] = f1_score(label_array, prediction_array, average='micro')


    auc_value,fprs, tprs, roc_auc, _, thresholds_optimal = multi_label_roc(epoch, test_labels, test_predictions, args.num_classes)
    fpr_macro = fprs["macro"]
    tpr_macro = tprs["macro"]
    roc_auc_macro = roc_auc["macro"]
    np.save('./inference/fpr_macro.npy', fpr_macro)
    np.save('./inference/tpr_macro.npy', tpr_macro)
    np.save('./inference/roc_auc_macro.npy', roc_auc_macro)


    pr_curves = get_precision_recall(test_labels, test_predictions)
    display_PRcurve(epoch, pr_curves, args.num_classes)
    recall_macro = pr_curves["macro"]["recall"]
    precision_macro = pr_curves["macro"]["precision"]
    pr_auc_macro = pr_curves["macro"]["average_precision"]
    np.save('./inference/recall_macro.npy', recall_macro)
    np.save('./inference/precision_macro.npy', precision_macro)
    np.save('./inference/pr_auc_macro.npy', pr_auc_macro)

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

def disp_confusion_matrix(cm, save_name, classes, title):
    disp = ConfusionMatrixDisplay(cm, display_labels=np.array(classes))
    disp.plot(cmap='GnBu')
    # plt.title(title)
    plt.savefig(save_name, format='png')
    # plt.show()


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
    # Compute micro-average ROC curve and ROC area（method1）
    fprs["micro"], tprs["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fprs["micro"], tprs["micro"])

    # Compute macro-average ROC curve and ROC area（method2）
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
    print("auc_macro:", roc_auc["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(dpi=1000)

    plt.plot(fprs["macro"], tprs["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    cls = {0: 'HP', 1: 'SSL', 2: 'TSA', 3: 'TA', 4: 'VTA', 5: 'VA'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta', 'blueviolet', 'seagreen'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fprs[i], tprs[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.3f})'
                       ''.format(cls[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('The Receiver Operating Characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('inference/{}_ROC.png'.format(epoch))
    # plt.show()
    # --------------------------------------------------------------------------------------------------------------#

    return aucs, fprs, tprs, roc_auc, thresholds, thresholds_optimal

def plot_confusion_matrix(cm, savename, classes, title):
    plt.figure(figsize=(12, 8), dpi=777)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Ground truth diagnoses')
    plt.xlabel('Predicted diagnoses')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')

def get_precision_recall(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    calculates precision-recall curve data from y true and prediction scores
    includes precision, recall, f1_score, thresholds, average_precision
    at each level of y, micro and macro averaged
    Args:
        y_true: true y values
        y_score: y prediction scores
    Returns:
        dict with precision-recall curve data
    """
    n_classes = y_true.shape[1]

    # Compute PR curve and average precision for each class
    pr_curves = {}
    for i in range(n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true[:, i], y_score[:, i])
        precision = np.delete(precision, -1)
        recall = np.delete(recall, -1)
        pr_curves[i] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                        "average_precision": metrics.average_precision_score(y_true[:, i], y_score[:, i])}

    # Compute micro-average PR curve and average precision
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.ravel(), y_score.ravel())
    precision = np.delete(precision, -1)
    recall = np.delete(recall, -1)
    pr_curves["micro"] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                          "average_precision": metrics.average_precision_score(y_true, y_score, "micro")}

    # Compute macro-average PR curve and average precision
    # First aggregate all false positive rates
    all_recall = np.unique(np.concatenate([pr_curves[i]["recall"] for i in range(n_classes)]))
    # Then interpolate all PR curves at this points
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        # xp needs to be increasing, but recall is decreasing, hence reverse the arrays
        mean_precision += interp(all_recall, pr_curves[i]["recall"][::-1], pr_curves[i]["precision"][::-1])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    # reverse the arrays back
    all_recall = all_recall[::-1]
    mean_precision = mean_precision[::-1]
    pr_curves["macro"] = {"precision": mean_precision, "recall": all_recall,
                          "average_precision": metrics.average_precision_score(y_true, y_score, "macro")}

    # calculate f1 score
    for i in pr_curves:
        precision = pr_curves[i]["precision"]
        recall = pr_curves[i]["recall"]
        pr_curves[i]["f1_score"] = 2 * (precision * recall) / (precision + recall)

    return pr_curves

def display_PRcurve(epoch, pr_curves, n_classes):
    lw = 2
    plt.figure(figsize=(12, 8), dpi=777)

    plt.plot(pr_curves["macro"]["recall"], pr_curves["macro"]["precision"],
             label='macro-average PR curve (area = {0:0.3f})'
                   ''.format(pr_curves["macro"]["average_precision"]),
             color='navy', linestyle=':', linewidth=3)


    cls = {0: 'HP', 1: 'SSL', 2: 'TSA', 3: 'TA', 4: 'VTA', 5: 'VA'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta', 'blueviolet', 'seagreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(pr_curves[i]["recall"], pr_curves[i]["precision"], color=color, lw=lw,
                 label='PR curve of class {0} (area = {1:0.3f})'
                       ''.format(cls[i], pr_curves[i]["average_precision"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('inference/{}_PRcurve.png'.format(epoch))
    # plt.show()

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Test MIST')
    parser.add_argument('--num_classes', default=6, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(1,), help='GPU ID(s) [0]')
    parser.add_argument('--dataset', default='Ade_val', type=str, help='Dataset folder name')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--model_path', default='', type=str, help='Path to save the model')
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
    state_dict_weights = torch.load(args.model_path)
    milnet.load_state_dict(state_dict_weights, strict=True)

    bags_csv = os.path.join('datasets', args.dataset, args.dataset + '.csv')

    bags_path = pd.read_csv(bags_csv)

    accuracy, F1score, test_loss_bag, avg_score, aucs, thresholds_optimal = test('test', bags_path, milnet, args)
    print(
        '\r accuracy: %.4f, F1score_macro: %.4f, F1score_micro: %.4f, average score: %.4f, AUC: ' %
        (accuracy, F1score["macro"], F1score["micro"],
         avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))




if __name__ == '__main__':
    main()