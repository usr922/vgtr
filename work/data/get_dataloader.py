# -*- coding: utf-8 -*-
from .unified_dataset import UnifiedDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader


input_transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


def get_train_loader(args):

    train_dataset = UnifiedDataset(data_root=args.data_root,
                                 split_root=args.split_root,
                                 dataset=args.dataset,
                                 split='train',
                                 imsize=args.size,
                                 transform=input_transform,
                                 max_query_len=args.max_query_len,
                                 augment=True)
    args.vocab_size = len(train_dataset.corpus)

    return DataLoader(train_dataset, batch_size=args.batch_size,
                      shuffle=True, pin_memory=True, drop_last=True,
                      num_workers=args.workers)


def get_val_loader(args):

    val_dataset = UnifiedDataset(data_root=args.data_root,
                                 split_root=args.split_root,
                                 dataset=args.dataset,
                                 split='val',
                                 imsize=args.size,
                                 transform=input_transform,
                                 max_query_len=args.max_query_len)

    return DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                      pin_memory=True, drop_last=True, num_workers=args.workers)


def get_test_loader(args, split):

    if args.dataset == 'refcoco' or args.dataset == 'refcoco+':
        assert split == 'testA' or split == 'testB'
    elif args.dataset == 'refcocog':
        assert split == 'val'
    else:
        assert split == 'test'

    test_dataset = UnifiedDataset(data_root=args.data_root,
                                  split_root=args.split_root,
                                  dataset=args.dataset,
                                  testmode=True,
                                  split=split,
                                  imsize=args.size,
                                  transform=input_transform,
                                  max_query_len=args.max_query_len)
    args.vocab_size = len(test_dataset.corpus)

    return DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                      pin_memory=True, drop_last=False, num_workers=args.workers)
