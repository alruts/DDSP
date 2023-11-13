import torch
import unittest

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.losses import *

class TestSpectralLoss(unittest.TestCase):
    def test_loss_shape(self):
        loss = SpectralLoss()
        target_audio = torch.randn(1, 1, 16000)  # Example target audio
        audio = torch.randn(1, 1, 16000)  # Example audio
        output = loss(target_audio, audio)
        # self.assertEqual(output.shape, torch.Size([]))

    def test_loss_with_weights(self):
        loss = SpectralLoss()
        target_audio = torch.randn(1, 1, 16000)  # Example target audio
        audio = torch.randn(1, 1, 16000)  # Example audio
        weights = torch.rand(1, 1, 256, 100)  # Example weight matrix
        output = loss(target_audio, audio, weights)
        # self.assertEqual(output.shape, torch.Size([]))

    def test_loss_type(self):
        loss = SpectralLoss(loss_type='L2')
        target_audio = torch.randn(1, 1, 16000)  # Example target audio
        audio = torch.randn(1, 1, 16000)  # Example audio
        output = loss(target_audio, audio)
        # self.assertEqual(output.shape, torch.Size([]))

    def test_loudness_weight(self):
        loss = SpectralLoss(loudness_weight=1.0)
        target_audio = torch.randn(1, 1, 16000)  # Example target audio
        audio = torch.randn(1, 1, 16000)  # Example audio
        output = loss(target_audio, audio)
        # self.assertEqual(output.shape, torch.Size([]))

    def test_custom_compute_loudness(self):
        # Implement a custom compute_loudness method for testing
        def custom_compute_loudness(audio, n_fft):
            return torch.randn(1, 1, 256, 100)  # Example loudness

        SpectralLoss.compute_loudness = staticmethod(custom_compute_loudness)
        loss = SpectralLoss(loudness_weight=1.0)
        target_audio = torch.randn(1, 1, 16000)  # Example target audio
        audio = torch.randn(1, 1, 16000)  # Example audio
        output = loss(target_audio, audio)
        # self.assertEqual(output.shape, torch.Size([]))

if __name__ == '__main__':
    unittest.main()
