import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from model import alexnet

from au_data_loader import au_data_loader
from helper import *

parser = argparse.ArgumentParser(description='AU Recognition')
parser.add_argument('--data_path_dir', default=r'E:\DataSets\CKPlus\cohn-kanade-images',
                    metavar='DIR', help='path to data dir')
parser.add_argument('--label_path_dir', default=r'E:\DataSets\CKPlus\FACS_labels\FACS',
                    metavar='DIR', help='path to label dir')
parser.add_argument('--landmark_path_dir', default=r'E:\DataSets\CKPlus\Landmarks\Landmarks',
                    metavar='DIR', help='path to landmark dir')
parser.add_argument('--emotion_path_dir', default=r'E:\DataSets\CKPlus\Emotion_labels\Emotion',
                    metavar='DIR', help='path to emotion dir')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0


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
        f1 = f1_score(output.data, target)
        f1_sc.update(f1, input.size(0))
        losses.update(loss.data[0], input.size(0))
        prec.update(acc, input.size(0))

        loss.backward()
        optimizer.step()
        if (i + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F1_Score {3} ({4})'.format(
                epoch, i + 1, len(train_loader), f1_sc.val, f1_sc.avg, loss=losses))


def valid(val_loader, model, criterion, print_freq=1):
    losses = AverageMeter()
    prec = AverageMeterList(10)
    f1_sc = AverageMeterList(10)
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        # with torch.no_grad():

        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)
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
        f1 = f1_score(output.data, target)
        losses.update(loss.data[0], input.size(0))
        prec.update(acc, input.size(0))
        f1_sc.update(f1, input.size(0))
        if (i + 1) % print_freq == 0:
            print('Validate: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  # 'Accuracy {2} ({3})\t'
                  'F1 {2} ({3})'.format(
                i + 1, len(val_loader), f1_sc.val, f1_sc.avg, loss=losses))
    return np.mean(prec.avg)


def main():
    global best_prec, args
    args = parser.parse_args()
    data_path_dir = args.data_path_dir  # r'E:\DataSets\CKPlus\cohn-kanade-images'
    label_path_dir = args.label_path_dir  # r'E:\DataSets\CKPlus\FACS_labels\FACS'
    landmark_path_dir = args.landmark_path_dir  # r'E:\DataSets\CKPlus\Landmarks\Landmarks'
    emotion_path_dir = args.emotion_path_dir  # r'E:\DataSets\CKPlus\Emotion_labels\Emotion'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = au_data_loader(data_path_dir, label_path_dir,
                                   landmark_path_dir, emotion_path_dir,
                                   dataset='train', transform=transform)
    valid_dataset = au_data_loader(data_path_dir, label_path_dir,
                                   landmark_path_dir, emotion_path_dir,
                                   dataset='valid', transform=transform)
    test_dataset = au_data_loader(data_path_dir, label_path_dir,
                                  landmark_path_dir, emotion_path_dir,
                                  dataset='test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # build model
    model = alexnet(pretrained=True)
    model.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    for param in model.features.parameters():
        param.requires_grad = False

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        valid(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch)
        prec = valid(val_loader, model, criterion)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict()
        }, is_best)


if __name__ == '__main__':
    main()
