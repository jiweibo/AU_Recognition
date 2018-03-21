import argparse
import itertools

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

from au_data_loader import *
from helper import *
from Model import *

# <editor-fold desc="settings">
parser = argparse.ArgumentParser(description='AU Recognition')
parser.add_argument('data_path_dir', default=r'E:\DataSets\CKPlus\cohn-kanade-images',
                    metavar='DIR', help='path to data dir')
parser.add_argument('label_path_dir', default=r'E:\DataSets\CKPlus\FACS_labels\FACS',
                    metavar='DIR', help='path to label dir')
parser.add_argument('landmark_path_dir', default=r'E:\DataSets\CKPlus\Landmarks\Landmarks',
                    metavar='DIR', help='path to landmark dir')
# parser.add_argument('--emotion_path_dir', default=r'E:\DataSets\CKPlus\Emotion_labels\Emotion',
#                     metavar='DIR', help='path to emotion dir')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = np.inf


# </editor-fold>


def train(train_loader, model, criterion, optimizer, epoch, print_freq=5):
    losses = AverageMeter()
    prec = AverageMeterList(10)
    f1_sc = AverageMeterList(10)
    model.train()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)
        optimizer.zero_grad()

        # compute output
        output = model(input_var)
        loss0 = criterion(output[:, 0], target_var[:, 0])
        loss1 = criterion(output[:, 1], target_var[:, 1])
        loss2 = criterion(output[:, 2], target_var[:, 2])
        loss3 = criterion(output[:, 3], target_var[:, 3])
        loss4 = criterion(output[:, 4], target_var[:, 4])
        loss5 = criterion(output[:, 5], target_var[:, 5])
        loss6 = criterion(output[:, 6], target_var[:, 6])
        loss7 = criterion(output[:, 7], target_var[:, 7])
        loss8 = criterion(output[:, 8], target_var[:, 8])
        loss9 = criterion(output[:, 9], target_var[:, 9])
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9
        acc = accuracy(output.data, target)
        # f1 = f1_score(output.data, target)
        # f1_sc.update(f1, input.size(0))
        losses.update(loss.data[0], input.size(0))
        prec.update(acc, input.size(0))

        loss.backward()
        optimizer.step()
        if (i + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                # 'F1_Score [{3}] ({4})'
                epoch, i + 1, len(train_loader), loss=losses))


def valid(val_loader, model, criterion, print_freq=1):
    losses = AverageMeter()
    prec = AverageMeterList(10)
    f1_sc = AverageMeterList(10)
    model.eval()
    return_pred, return_tar = [], []

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(async=True)
        # with torch.no_grad():
        input_var = Variable(input, volatile=True)
        target_var = Variable(target.cuda(async=True), volatile=True)
        # compute output
        output = model(input_var)
        loss0 = criterion(output[:, 0], target_var[:, 0])
        loss1 = criterion(output[:, 1], target_var[:, 1])
        loss2 = criterion(output[:, 2], target_var[:, 2])
        loss3 = criterion(output[:, 3], target_var[:, 3])
        loss4 = criterion(output[:, 4], target_var[:, 4])
        loss5 = criterion(output[:, 5], target_var[:, 5])
        loss6 = criterion(output[:, 6], target_var[:, 6])
        loss7 = criterion(output[:, 7], target_var[:, 7])
        loss8 = criterion(output[:, 8], target_var[:, 8])
        loss9 = criterion(output[:, 9], target_var[:, 9])
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9

        return_pred.extend(output.data.cpu().tolist())
        return_tar.extend(target.tolist())
        # acc = accuracy(output.data, target)
        # f1 = f1_score(output.data, target)
        losses.update(loss.data[0], input.size(0))
        # prec.update(acc, input.size(0))
        # f1_sc.update(f1, input.size(0))
        if (i + 1) % print_freq == 0:
            print('Validate: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                # 'F1_Score [{2}] ({3})'
                i + 1, len(val_loader), loss=losses))

    return return_tar, return_pred, np.mean(losses.avg)


def main():
    global best_prec, args
    args = parser.parse_args()
    data_path_dir = args.data_path_dir  # r'E:\DataSets\CKPlus\cohn-kanade-images'
    label_path_dir = args.label_path_dir  # r'E:\DataSets\CKPlus\FACS_labels\FACS'
    landmark_path_dir = args.landmark_path_dir  # r'E:\DataSets\CKPlus\Landmarks\Landmarks'
    # emotion_path_dir = args.emotion_path_dir  # r'E:\DataSets\CKPlus\Emotion_labels\Emotion'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    reserved_set, reserved_label = get_reserved_set(label_path_dir)
    au_image = load_au_image_from_path(data_path_dir)
    au_label = load_au_label_from_path(label_path_dir, reserved_label, reserved_set)
    au_landmark = load_au_landmark_from_path(landmark_path_dir)
    for i in range(len(au_image)):
        au_image[i] = np.array(crop_au_img(au_image[i], au_landmark[i]))
    au_image = np.array(au_image)
    au_label = np.array(au_label)

    # build model
    model, criterion, optimizer = build_model()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    fold = 5
    kf = KFold(fold, shuffle=True, random_state=20)
    res_tar, res_pred = [], []
    for k, (train_index, test_index) in enumerate(kf.split(au_image, au_label)):
        train_dataset = au_data_loader(au_image[train_index], au_label[train_index], transform=transform)
        valid_dataset = au_data_loader(au_image[test_index], au_label[test_index], transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        if args.evaluate:
            tar, pred, _ = valid(valid_loader, model, criterion)
            res_pred.extend(pred)
            res_tar.extend(tar)
            continue

        # build a new model
        model, criterion, optimizer = build_model()

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr, 10)
            train(train_loader, model, criterion, optimizer, epoch)
        tar, pred, ls = valid(valid_loader, model, criterion)
        res_pred.extend(pred)
        res_tar.extend(tar)

        is_best = ls < best_prec
        best_prec = min(ls, best_prec)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict()
        }, is_best)
        print('fold: {0}\t loss: {1}'.format(k, ls))

    res_pred = np.array(res_pred)
    res_tar = np.array(res_tar)

    out = []
    for i in range(res_tar.shape[1]):
        print()
        print('AU' + str(list(reserved_set)[i]) + ':' +
              str(f1_score(res_tar[:, i], np.around(res_pred[:, i]))))
        out.append('AU' + str(list(reserved_set)[i]) + ':' +
              str(f1_score(res_tar[:, i], np.around(res_pred[:, i]))))
        cm = confusion_matrix(res_tar[:, i], np.around(res_pred[:, i]))
        plt.figure()
        plot_confusion_matrix(cm, classes=[0, 1])
        # plt.figure()
        # plot_confusion_matrix(cm, classes=[0, 1], normalize=True)
        plt.show()
        print()

    # write to txt
    with open('alexnet_output.txt', 'w') as f:
        for i in out:
            f.writelines(i)
            f.write('\n')


def build_model(pretrained=True):
    model = alexnet(pretrained=pretrained)
    model.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    for param in model.features.parameters():
        param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = False
    return model, criterion, optimizer


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()
