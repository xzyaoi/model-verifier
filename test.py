import unittest
from unittest import TestCase
from zonotope import Zonotope
import torch

class TestZonotope(TestCase):

    def test_bound(self):
        testtope = Zonotope(1, torch.Tensor([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]))
        self.assertEqual(testtope.get_bound(), (0.4, 1.6))

    def test_relu(self):
        testtope = Zonotope(1, torch.Tensor([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]))
        self.assertEqual(testtope.relu(), 1, [0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0])
        testtope = Zonotope(-1, torch.Tensor([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]))
        self.assertEqual(testtope.relu(), 0, [0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0])


if __name__ == '__main__':
    unittest.main()