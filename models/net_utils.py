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
    print(f"threshold: {threshold}")
    '''The network outputs a binary probability. For the final result and some metrics, like the jackard score, we need to decide if the value is 1 or 0.'''
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image

def get_binary_data(masks, images, threshold = 0.5):
    # I know bad code...
    try:
        masks = masks.detach().cpu().numpy()
        images = images.detach().cpu().numpy()
    except:
        pass

    masks = binarize_image_pp(masks, threshold)
    images = binarize_image_pp(images, threshold)

    return masks, images

def calculate_scores(masks, images, score_fct, thresholds):
    scores = []
    for threshold in thresholds:
        masks, images = get_binary_data(masks, images, threshold)

        scores.append(score_fct(masks.flatten(), images.flatten()))
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

# Binary CNN specific functions
def calculate_correctness_for_binary_input(pred, mask):
    assert mask in [0, 1]
    assert pred <= 1 and pred >= 0

    correct_positive = 0
    correct_negative = 0
    false_positive = 0
    false_negative = 0

    if mask == 1:
        if pred > 0.5:
            correct_positive = 1
        else:
            false_negative = 1
    else:
        if pred < 0.5:
            correct_negative = 1
        else:
            false_positive = 1
    
    return np.array([correct_positive, correct_negative, false_positive, false_negative])
    