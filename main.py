import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from model import alexnet

from PIL import Image
from au_data_loader import au_data_loader

data_path_dir = r'E:\DataSets\CKPlus\cohn-kanade-images'
label_path_dir = r'E:\DataSets\CKPlus\FACS_labels\FACS'
landmark_path_dir = r'E:\DataSets\CKPlus\Landmarks\Landmarks'
emotion_path_dir = r'E:\DataSets\CKPlus\Emotion_labels\Emotion'

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
                              dataset='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

# build model
model = alexnet(pretrained=True)
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
for param in model.features.parameters():
    param.requires_grad = False


def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = Variable(input)
        # target_var = Variable(target)

        # compute output
        output = model(input_var)
        # loss1 = criterion(output, target_var)


def valid(val_loader):
    pass


def test(test_loader):
    pass


def main():
    train(train_loader, model, criterion, optimizer, 0, 1)


if __name__ == '__main__':
    main()
