import torch
import numpy as np

def get_weighted_bce_loss(pred, mask):
    # the positive class (mask == 1) has a much higher weight if missed.
    class_weight = torch.tensor([0.5, 0.5])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight[1])

    return loss_fn(pred, mask)

# Jaccard Index (IoU)
# meant to capture how much the prediction class overlaps with the actual image.
def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    
    return loss.mean()