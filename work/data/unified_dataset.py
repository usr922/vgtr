# -*- coding: utf-8 -*-

import sys
import cv2
import torch
import numpy as np
import os.path as osp
import torch.utils.data as data
from .. import utils
from ..utils import Corpus
from ..utils.transforms import trans, trans_simple
sys.path.append('.')
sys.modules['utils'] = utils
cv2.setNumThreads(0)

class UnifiedDataset(data.Dataset):

    SUPPORTED_DATASETS = {
        'refcoco': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },

        'refcoco+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },

        'refcocog': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },

        'refcocog_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },

        'flickr': {
            'splits': ('train', 'val', 'test')
        },

        'copsref': {
            'splits': ('train', 'val', 'test')
        }
    }

    # map the dataset name to data folder
    MAPPING = {
        'refcoco': 'unc',
        'refcoco+': 'unc+',
        'refcocog': 'gref',
        'refcocog_umd': 'gref_umd',
        'flickr': 'flickr',
        'copsref': 'copsref'
    }

    def __init__(self, data_root, split_root='data', dataset='refcoco', imsize=512, transform=None, testmode=False, split='train', max_query_len=20, augment=False):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.trans = trans if augment else trans_simple

        if self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        elif self.dataset == 'copsref':
            self.dataset_root = osp.join(self.data_root, 'copsref')
            self.im_dir = osp.join(self.dataset_root, 'images')
        else:
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')



        self.sup_set = self.dataset
        self.dataset = self.MAPPING[self.dataset]

        if not self.exists_dataset():
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.sup_set]['splits']

        self.corpus = Corpus()
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        # splits = [split]
        splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):

        return osp.exists(osp.join(self.split_root, self.dataset))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """

        :return: (img, phrase word id, phrase word mask, bounding bbox)
        """
        if self.dataset == 'flickr' or self.dataset == 'copsref':
            img_file, bbox, phrase = self.images[index]
        else:
            img_file, _, bbox, phrase, _ = self.images[index]

        if not self.dataset == 'flickr':
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)

        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)

        phrase = phrase.lower()

        img, phrase, bbox = self.trans(img, phrase, bbox, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        # tokenize phrase
        word_id = self.corpus.tokenize(phrase, self.query_len)
        word_mask = np.array(word_id > 0, dtype=int)

        return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
