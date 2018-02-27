import shutil
import torch
import numpy as np


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


class AverageMeterList(object):
    def __init__(self, length):
        self.val = [0] * length
        self.avg = [0] * length
        self.sum = [0] * length
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = (np.array(val) * n + np.array(self.sum)).tolist()
        self.count += n
        self.avg = (np.array(self.sum) / self.count).tolist()


def accuracy(output, target):
    pred = (output >= 0.5).float()
    acc = pred.eq(target).sum(dim=0).float().div(output.size(0)).mul_(100)
    return acc.tolist()


def f1_score(output, target, eps=1e-5):
    # todo: divide zero
    pred = (output >= 0.5).float()
    tp = (pred * target).sum(dim=0)
    p = pred.sum(dim=0)
    precision = tp / (p+eps)
    t = target.sum(dim=0)
    recall = tp / (t+eps)
    f1 = 2 * precision * recall / (precision + recall)
    return f1.cpu().tolist()


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='au_model.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'au_model_best.pth')
