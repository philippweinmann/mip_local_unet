import torch
import numpy as np

def dice_loss(y_pred, y_true, smooth=1e-7):
    intersection = torch.sum(y_true * y_pred, dim=(1,2,3))
    sum_of_squares_pred = torch.sum(torch.square(y_pred), dim=(1,2,3))
    sum_of_squares_true = torch.sum(torch.square(y_true), dim=(1,2,3))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    return dice.mean()

def get_weighted_bce_loss():
    # the positive class (mask == 1) has a much higher weight if missed.
    class_weight = torch.tensor([0.5, 0.5])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight[1])

    return loss_fn

# Jaccard Index (IoU)
# meant to capture how much the prediction class overlaps with the actual image.
def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)