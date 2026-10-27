import unittest
import torch
from PIL import Image
from src.policy import clipped_loss,standardized,color_fraction


class PolicyTests(unittest.TestCase):
    def test_clipping(self):
        loss=clipped_loss(torch.tensor([2.]).log(),torch.zeros(1),1.,.2)
        self.assertAlmostEqual(loss.item(),-1.2,places=5)
        self.assertTrue((standardized([4,4])==0).all())

    def test_independent_color(self):
        self.assertEqual(color_fraction(Image.new("RGB",(6,6),"red"),0),1.)
        self.assertEqual(color_fraction(Image.new("RGB",(6,6),"red"),2),0.)
