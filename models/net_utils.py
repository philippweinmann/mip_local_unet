from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score # this is the dice score
from scipy.spatial.distance import directed_hausdorff

import torch
import numpy as np

import time

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
    # I know bad code...
    try:
        image = image.detach().cpu().numpy()
    except:
        pass

    image = np.squeeze(image)

    return image

def binarize_image_pp(image, threshold = 0.5):
    '''The network outputs a binary probability. For the final result and some metrics, like the jackard score, we need to decide if the value is 1 or 0.'''
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image

def get_binary_data(masks, images, threshold = 0.5):
    masks = binarize_image_pp(masks, threshold)
    images = binarize_image_pp(images, threshold)

    return masks, images

def calculate_scores(masks: np.array, images: np.array, score_fct, thresholds):
    scores = []
    
    flattened_masks = masks.flatten()
    flattened_images = images.flatten()
    
    for threshold in thresholds:
        binary_masks, binary_images = get_binary_data(flattened_masks, flattened_images, threshold)
        scores.append(score_fct(binary_masks, binary_images))
        
    return scores

def calculate_jaccard_score(masks, images, thresholds):
    '''
    0: no overlap
    1: perfect overlap
    '''
    return calculate_scores(masks, images, score_fct=jaccard_score, thresholds=thresholds)

def calculate_dice_scores(masks, images, thresholds):
    '''
    0: no overlap
    1: perfect overlap
    '''
    return calculate_scores(masks, images, score_fct=f1_score, thresholds=thresholds)

def calculate_hausdorff_distance(masks, images):
    '''
    0: no overlap
    1: perfect overlap
    '''
    # todo fix for 3d data
    masks, images = get_binary_data(masks, images)

    score = directed_hausdorff(masks, images)
    return score

def calculate_overlap(masks, images, thresholds):
    scores = []
    
    flattened_masks = masks.flatten()
    flattened_images = images.flatten()
    
    for threshold in thresholds:
        binary_masks, binary_images = get_binary_data(flattened_masks, flattened_images, threshold)
        
        mask_count = np.sum(binary_masks)
        
        # Calculate overlap
        overlap = np.logical_and(binary_masks, binary_images)

        # Count the number of overlapping pixels
        overlap_count = np.sum(overlap)
        normalized_overlap = overlap_count / mask_count if mask_count > 0 else 0
        
        scores.append(normalized_overlap)
        
    return scores

def get_best_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device

def save_model(model):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    model_name = "3d_model" + timestr + ".pth"
    model_save_path = "saved_models/" + model_name

    # saving the model
    torch.save(model.state_dict(), model_save_path)
    print(f"model saved at: {model_save_path}")

max_dice_threshold = 20000
total_weight = 1.5
min_bce_weight = 0.2

def get_appropriate_dice_weight(amt_positive_voxels):
    bce_weight = (-(total_weight - min_bce_weight)/max_dice_threshold) * amt_positive_voxels + 1.5
    bce_weight = max(bce_weight, min_bce_weight)
    
    dice_weight = total_weight - bce_weight

    return dice_weight, bce_weight


pos_voxel_threshold = 7000
max_lr_threshold = 0.1

def calculate_learning_rate(amt_positive_voxels, epoch):
    # make the dice loss an exponential function. 0.0001 if there are no pos voxels, 0.1 if above pos_voxel_threshold
    lr = 10 ** -(epoch) * 0.0001 * np.exp(amt_positive_voxels * (3*np.log(10)/pos_voxel_threshold))
    lr = min(lr, max_lr_threshold)

    # print(f"learning rate: {lr}, amt_positive_voxels: {amt_positive_voxels}")
    return lr