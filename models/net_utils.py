from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score # this is the dice score
from scipy.spatial.distance import directed_hausdorff

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

def get_binary_data(masks, images):
    try:
        masks = masks.detach().cpu().numpy()
        images = images.detach().cpu().numpy()
    except:
        pass

    masks = binarize_image_pp(masks)
    images = binarize_image_pp(images)

    return masks, images

def calculate_score(masks, images, score_fct):
    masks, images = get_binary_data(masks, images)

    score = score_fct(masks.flatten(), images.flatten())
    return score

def calculate_jaccard_score(masks, images):
    '''
    0: no overlap
    1: perfect overlap
    '''
    return calculate_score(masks, images, score_fct=jaccard_score)

def calculate_dice_score(masks, images):
    '''
    0: no overlap
    1: perfect overlap
    '''
    return calculate_score(masks, images, score_fct=f1_score)

def calculate_hausdorff_distance(masks, images):
    '''
    0: no overlap
    1: perfect overlap
    '''
    # todo fix for 3d data
    masks, images = get_binary_data(masks, images)

    score = directed_hausdorff(masks, images)
    return score


def get_best_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device