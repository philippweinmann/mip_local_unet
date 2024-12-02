import unittest
from models.unet3D import UNet3D, dice_bce_loss, softdiceloss
from models.net_utils import get_best_device, prepare_image_for_network_input, calculate_learning_rate

import numpy as np

class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        self.model = UNet3D(in_channels=1, num_classes=1)
        self.device = get_best_device()
        self.model.to(self.device)

        patch_size = 128
        patch_shape = (patch_size, patch_size, patch_size)
        self.empty_input = prepare_image_for_network_input(np.zeros(patch_shape)).to(self.device)
        self.full_input = prepare_image_for_network_input(np.ones(patch_shape)).to(self.device)
        self.empty_target = prepare_image_for_network_input(np.zeros(patch_shape)).to(self.device)
        
        # half empty image: half 1s and half 0s
        self.half_empty_input = prepare_image_for_network_input(np.concatenate([np.zeros((64, 128, 128)), np.ones((64, 128, 128))])).to(self.device)
        return super().setUp()

    def test_model_works(self):
        pred_empty = self.model(self.empty_input)
        pred_full = self.model(self.full_input)

    def test_soft_dice_loss(self):
        loss = softdiceloss(predictions=self.empty_input, targets=self.empty_target)
        self.assertEqual(loss, 0)

        loss = softdiceloss(predictions=self.full_input, targets=self.full_input)
        self.assertEqual(loss, 0)

        loss = softdiceloss(predictions=self.full_input, targets=self.empty_input)
        self.assertEqual(loss, 1)

        loss = softdiceloss(predictions=self.half_empty_input, targets=self.full_input)
        self.assertTrue(loss - 1/3 < 0.001)
    
    def test_dice_bce_loss(self):
        weights = (1, 0) # full dice loss
        loss = dice_bce_loss(predictions=self.empty_input, targets=self.empty_target, weights=weights)
        self.assertEqual(loss, 0)

        loss = dice_bce_loss(predictions=self.full_input, targets=self.full_input, weights=weights)
        self.assertEqual(loss, 0)

        loss = dice_bce_loss(predictions=self.empty_input, targets=self.empty_input, weights=weights)
        self.assertEqual(loss, 0)

        loss = dice_bce_loss(predictions=self.full_input, targets=self.empty_input, weights=weights)
        self.assertEqual(loss, 1)

        loss = dice_bce_loss(predictions=self.half_empty_input, targets=self.full_input, weights=weights)
        self.assertTrue(loss - 1/3 < 0.001)

        weights = (0, 1) # full bce loss
        loss = dice_bce_loss(predictions=self.empty_input, targets=self.empty_target, weights=weights)
        self.assertEqual(loss, 0)

        loss = dice_bce_loss(predictions=self.full_input, targets=self.full_input, weights=weights)
        self.assertEqual(loss, 0)

        loss = dice_bce_loss(predictions=self.empty_input, targets=self.empty_input, weights=weights)
        self.assertEqual(loss, 0)

        loss = dice_bce_loss(predictions=self.full_input, targets=self.empty_input, weights=weights)
        self.assertEqual(loss, 1) # that is the bce loss for fully wrong predictions

        loss = dice_bce_loss(predictions=self.half_empty_input, targets=self.full_input, weights=weights)
        self.assertTrue(loss, 0.5)

    def test_dynamic_lr(self):
        epoch = 0
        amt_positive_voxels = 0
        lr = calculate_learning_rate(amt_positive_voxels, epoch)
        assert lr == 0.00001

        epoch = 0
        amt_positive_voxels = 7000
        lr = calculate_learning_rate(amt_positive_voxels, epoch)
        assert lr - 0.01 <= 0.000000000001

        epoch = 0
        amt_positive_voxels = 200000
        lr = calculate_learning_rate(amt_positive_voxels, epoch)
        print(lr)

if __name__ == '__main__':
    unittest.main()