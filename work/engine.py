# -*- coding: utf-8 -*-

import time
import logging
import numpy as np
import torch
from torch.autograd import Variable
from .utils.utils import AverageMeter, xywh2xyxy, bbox_iou


def train_epoch(args, train_loader, model, optimizer, epoch, criterion=None, img_size=512):

    batch_time = AverageMeter()
    losses = AverageMeter()

    losses_bbox = AverageMeter()
    losses_giou = AverageMeter()

    acc = AverageMeter()
    miou = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(train_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        bbox = bbox.cuda()
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)
        image = Variable(imgs)
        word_id = Variable(word_id)
        bbox = Variable(bbox)

        norm_bbox = torch.zeros_like(bbox).cuda()

        norm_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]   # w
        norm_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]    # h

        # forward
        pred_box = model(image, word_id)  # [bs, C, H, W]
        loss, loss_box, loss_giou = criterion(pred_box, norm_bbox, img_size=img_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pred-box
        pred_bbox = pred_box.detach().cpu()
        pred_bbox = pred_bbox * img_size
        pred_box = xywh2xyxy(pred_bbox)

        losses.update(loss.item(), imgs.size(0))
        losses_bbox.update(loss_box.item(), imgs.size(0))
        losses_giou.update(loss_giou.item(), imgs.size(0))

        target_bbox = bbox
        iou = bbox_iou(pred_box, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        # metrics
        miou.update(torch.mean(iou).item(), imgs.size(0))
        acc.update(accu, imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Loss_bbox {loss_box.val:.4f} ({loss_box.avg:.4f})\t' \
                        'Loss_giou {loss_giou.val:.4f} ({loss_giou.avg:.4f})\t' \
                        'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                .format(epoch+1, batch_idx+1, len(train_loader),
                        batch_time=batch_time,
                        loss=losses,
                        loss_box=losses_bbox,
                        loss_giou=losses_giou,
                        acc=acc,
                        miou=miou)

            print(print_str)
            logging.info(print_str)


def validate_epoch(args, val_loader, model, train_epoch, img_size=512):

    batch_time = AverageMeter()
    acc = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size-1)

        norm_bbox = torch.zeros_like(bbox).cuda()

        norm_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]   # w
        norm_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]    # h

        with torch.no_grad():
            pred_box = model(image, word_id)  # [bs, C, H, W]

        pred_bbox = pred_box.detach().cpu()
        pred_bbox = pred_bbox * img_size
        pred_bbox = xywh2xyxy(pred_bbox)

        # constrain
        pred_bbox[pred_bbox < 0.0] = 0.0
        pred_bbox[pred_bbox > img_size-1] = img_size-1

        target_bbox = bbox
        # metrics
        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        # accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / imgs.size(0)

        acc.update(accu, imgs.size(0))
        miou.update(torch.mean(iou).item(), imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % (args.print_freq//10) == 0:
            print_str = 'Validate: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'Acc {acc.val:.4f} ({acc.avg:.4f})  ' \
                        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})  ' \
                .format(batch_idx+1, len(val_loader), batch_time=batch_time, acc=acc, miou=miou)

            print(print_str)
            logging.info(print_str)

    print(f"Train_epoch {train_epoch+1}  Validate Result:  Acc {acc.avg}, MIoU {miou.avg}.")

    logging.info("Validate: %f, %f" % (acc.avg, float(miou.avg)))

    return acc.avg, miou.avg

def test_epoch(test_loader, model, img_size=512):

    acc = AverageMeter()
    miou = AverageMeter()
    model.eval()

    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(test_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=img_size-1)

        norm_bbox = torch.zeros_like(bbox).cuda()

        norm_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]   # w
        norm_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]    # h

        with torch.no_grad():
            pred_box = model(image, word_id)  # [bs, C, H, W]

        pred_bbox = pred_box.detach().cpu()
        pred_bbox = pred_bbox * img_size
        pred_bbox = xywh2xyxy(pred_bbox)

        # constrain
        pred_bbox[pred_bbox < 0.0] = 0.0
        pred_bbox[pred_bbox > img_size-1] = img_size-1

        target_bbox = bbox
        # metrics
        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / imgs.size(0)

        acc.update(accu, imgs.size(0))
        miou.update(torch.mean(iou).item(), imgs.size(0))

    print(f"Test Result:  Acc {acc.avg}, MIoU {miou.avg}.")
