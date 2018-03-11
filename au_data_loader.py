import os

import torch
from torch.utils.data import Dataset

from PIL import Image, ImageDraw
import numpy as np


def get_reserved_set(label_path_dir):
    au_label = []
    returned_label = []
    for names in os.listdir(label_path_dir):
        name = os.path.join(label_path_dir, names)
        for sequences in os.listdir(name):
            sequence = os.path.join(name, sequences)
            if os.listdir(sequence):
                temp = np.loadtxt(os.path.join(sequence, os.listdir(sequence)[-1]))
                if temp.ndim == 1:
                    au_label.append(temp.astype(np.int32))
                    returned_label.append(temp[0].astype(np.int32))
                elif temp.ndim == 2:
                    au_label.extend(temp.astype(np.int32))
                    returned_label.append(temp[:, 0].astype(np.int32))
                else:
                    print(temp)
                    raise Exception("label info error!")
    ll = []
    for au in au_label:
        if au.ndim == 2:
            ll.extend(au[:, 0].astype(np.int32))
        elif au.ndim == 1:
            ll.append(au[0].astype(np.int32))
        else:
            print(au)
            raise Exception("label info error!")
    label_set = set(ll)
    reserved_set = set()
    for label in label_set:
        if ll.count(label) > 90:
            reserved_set.add(label)
    return reserved_set, returned_label  # (1, 2, 4, 5, 6, 7, 12, 15, 17, 25)


def convert_label(init_label, reserved_set):
    reserved_list = list(reserved_set)
    converted_label = [0] * len(reserved_set)
    if type(init_label) == list:
        for l in init_label:
            for i in range(len(reserved_list)):
                if l == reserved_list[i]:
                    converted_label[i] = 1
    elif type(init_label) == int:
        for i in range(len(reserved_list)):
            if init_label == reserved_list[i]:
                converted_label[i] = 1
    else:
        print("init label is not a list! '", init_label, "'")
        print("type is", type(init_label))
        raise Exception

    return converted_label


def draw_landmark_point(img, landmark_path):
    img = Image.open(img)
    landmark = np.loadtxt(landmark_path)
    draw = ImageDraw.Draw(img)
    t = 1
    for point in landmark:
        draw.text(point.tolist(), str(t), fill=255)
        t += 1
    img.show()


def crop_au_img(img, landmark):
    width, height = img.size
    left = max(int(min(landmark[:, 0])) - 50, 0)
    right = min(width, int(max(landmark[:, 0] + 50)))
    top = max(int(min(landmark[:, 1])) - 100, 0)
    bottom = min(height, int(max(landmark[:, 1])) + 10)
    img = img.crop((left, top, right, bottom))
    return img


def load_au_image_from_path(data_path_dir):
    # prepare au image
    au_image = []
    for names in os.listdir(data_path_dir):
        name = os.path.join(data_path_dir, names)
        for sequences in os.listdir(name):
            sequence = os.path.join(name, sequences)
            if os.path.isdir(sequence):
                if os.listdir(sequence):
                    au_image.append(
                        Image.open(os.path.join(sequence, os.listdir(sequence)[-1])).convert('RGB'))
    return au_image


def load_au_label_from_path(label_path_dir, reserved_label, reserved_set):
    # prepare au label
    au_label = []
    for l in reserved_label:
        au_label.append(convert_label(l.tolist(), reserved_set))
    return au_label


def load_au_landmark_from_path(landmark_path_dir):
    au_landmark = []
    for names in os.listdir(landmark_path_dir):
        name = os.path.join(landmark_path_dir, names)
        for sequences in os.listdir(name):
            sequence = os.path.join(name, sequences)
            if os.listdir(sequence):
                au_landmark.append(np.loadtxt(os.path.join(sequence, os.listdir(sequence)[-1])))
    return au_landmark


def load_au_emotion_from_path(emotion_path_dir):
    au_emotion_landmark_path = []
    for names in os.listdir(emotion_path_dir):
        name = os.path.join(emotion_path_dir, names)
        for sequences in os.listdir(name):
            sequence = os.path.join(name, sequences)
            if os.listdir(sequence):
                au_emotion_landmark_path.append(os.path.join(sequence, os.listdir(sequence)[-1]))
    return au_emotion_landmark_path


class au_data_loader(Dataset):
    def __init__(self, au_image, au_label, transform=None, target_transform=None):
        # prepare au image
        self.au_image = au_image

        # prepare au label
        self.au_label = au_label
        self.au_label = torch.from_numpy(np.array(self.au_label)).float()

        # prepare au landmark
        # self.au_landmark = au_landmark
        # prepare au emotions
        # self.au_emotion_landmark_path = au_emotion_landmark_path

        self.transform = transform
        self.target_transform = target_transform

        self.train_data = self.au_image
        self.train_label = self.au_label

    def __getitem__(self, index):
        img = self.train_data[index]
        img = Image.fromarray(img)
        target = self.train_label[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)
