import torch
import numpy as np

def get_weighted_bce_loss(pred, mask):
    # the positive class (mask == 1) has a much higher weight if missed.
    class_weight = torch.tensor([0.5, 0.5])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight[1])

    return loss_fn(pred, mask)

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    
    return loss.mean()

def prepare_image_for_network_input(image):
    image = image[None, None, :, :]
    t_image = torch.tensor(image, dtype=torch.float32)

    return t_image


def prepare_image_for_analysis(image):
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)

    return image

def binarize_image_pp(image):
    '''The network outputs a binary probability. For the final result and some metrics, like the jackard score, we need to decide if the value is 1 or 0.'''
    binary_image = np.where(image > 0.5, 1, 0)
    return binary_image