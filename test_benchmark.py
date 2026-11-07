"""Regression checks for the benchmark's corruption operators."""
import unittest

import torch

from benchmark import corrupt, mixed_augmentation, shift_horizontal
from experiment import translate_right


class CorruptionTests(unittest.TestCase):
    def test_signed_shift_fills_exposed_pixels(self):
        images = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]]).repeat(2, 1, 1, 1)
        shifted = shift_horizontal(images, torch.tensor([1, -1]))
        expected = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]], [[[2.0, 3.0, 4.0, 0.0]]]])
        torch.testing.assert_close(shifted, expected)

    def test_three_pixel_shift_matches_original_benchmark(self):
        images = torch.arange(56.0).reshape(2, 1, 1, 28)
        torch.testing.assert_close(shift_horizontal(images, torch.tensor([3, 3])),
                                   translate_right(images))

    def test_blur_spreads_impulse_without_changing_shape(self):
        image = torch.zeros((1, 1, 9, 9))
        image[0, 0, 4, 4] = 1
        blurred = corrupt(image, "blur", 3, torch.Generator())
        expected = torch.zeros_like(image)
        expected[0, 0, 3:6, 3:6] = 1 / 9
        torch.testing.assert_close(blurred, expected)

    def test_mixed_augmentation_is_seeded_and_preserves_input(self):
        images = torch.full((24, 1, 28, 28), 0.5)
        first = mixed_augmentation(images, torch.Generator().manual_seed(11))
        second = mixed_augmentation(images, torch.Generator().manual_seed(11))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(images, torch.full_like(images, 0.5))
        self.assertTrue((first >= 0).all() and (first <= 1).all())


if __name__ == "__main__":
    unittest.main()
