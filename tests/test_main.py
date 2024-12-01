import unittest
from models.unet3D import UNet3D
from models.net_utils import get_best_device, prepare_image_for_network_input, prepare_image_for_analysis
import torch
import numpy as np

class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        self.model = UNet3D(in_channels=1, num_classes=1)
        self.device = get_best_device()
        self.model.to(self.device)

        patch_size = 128
        self.empty_input = prepare_image_for_network_input(np.zeros((patch_size, patch_size, patch_size))).to(self.device)
        self.full_input = prepare_image_for_network_input(np.ones((patch_size, patch_size, patch_size))).to(self.device)
        return super().setUp()

    def test_model(self):
        pred_empty = self.model(self.empty_input)
        pred_full = self.model(self.full_input)

if __name__ == '__main__':
    unittest.main()