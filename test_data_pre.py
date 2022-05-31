from unittest import TestCase

import torch

from data_pre import Batch

tgt = torch.arange(0,10).reshape(2,5)


c = Batch(None,None)

class TestBatch(TestCase):
    def test_make_std_mask(self):
        c.make_std_mask(tgt,0)
        # self.fail()
