# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added LMDB dataset -- Youwei Liang
import os
import json
import yaml

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch.utils.data as data

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from folder2lmdb import ImageFolderLMDB

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'cub': (0.3659524, 0.42010019, 0.41562049),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'cub': (0.07625843, 0.04599726, 0.06182727),
    'cifar10': (0.2470, 0.2435, 0.2616)
}

def get_set(args, is_train, transform=None):
    if args.dataset_name == 'cifar10':
        ds = datasets.CIFAR10(root=args.dataset_root_path,
                              train=is_train,
                              transform=transform, download=True)
        ds.num_classes = 10
    else:
        ds = DatasetImgTarget(args, is_train=is_train, transform=transform)
        args.num_classes = ds.num_classes

    if is_train:
        setattr(args, f'num_images_train', ds.__len__())
        print(f"{args.dataset_name} train split. N={ds.__len__()}, K={ds.num_classes}.")
    else:
        setattr(args, f'num_images_val', ds.__len__())
        print(f"{args.dataset_name} val split. N={ds.__len__()}, K={ds.num_classes}.")
    return ds

class DatasetImgTarget(data.Dataset):
    def __init__(self, args, is_train, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform

        if is_train==True:
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif is_train==False:
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'
        # assert os.path.isdir(os.path.join(self.root, self.images_folder)), \
        #    f'{os.path.join(self.root, self.images_folder)} is not a directory.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset_name == 'CIFAR':
        dataset = datasets.CIFAR100(args.dataset_root_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.dataset_name == 'IMNET':
        if args.use_lmdb:
            root = os.path.join(args.dataset_root_path, 'train.lmdb' if is_train else 'val.lmdb')
            if not os.path.isfile(root):
                raise FileNotFoundError(f"LMDB dataset '{root}' is not found. "
                        "Pleaes first build it by running 'folder2lmdb.py'.")
            dataset = ImageFolderLMDB(root, transform=transform)
        else:
            root = os.path.join(args.dataset_root_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.dataset_name == 'INAT':
        dataset = INatDataset(args.dataset_root_pathh, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.dataset_name == 'INAT19':
        dataset = INatDataset(args.dataset_root_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    else:
        dataset = DatasetImgTarget(args, is_train=is_train, transform=transform)
        nb_classes = dataset.num_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    input_size = args.input_size
    resize_size = int(args.input_size / 0.875)
    test_resize_size = resize_size

    mean = MEANS['imagenet']
    std = STDS['imagenet']
    if args.custom_mean_std:
        mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
        std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

    t = []

    if is_train:
        t.append(transforms.Resize(
            (resize_size, resize_size),
            interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.RandomCrop(input_size))
        t.append(transforms.RandomHorizontalFlip())
    else:
        t.append(transforms.Resize(
            (test_resize_size, test_resize_size),
            interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(t)
    print(transform)
    return transform
