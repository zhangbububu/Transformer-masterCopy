import copy
from collections import Counter

import numpy as np
import torch

import torch
import torch.nn as nn
import math

from model import LabelSmoothing

criterion = LabelSmoothing(3,padding_idx=0, smoothing=0.0)

import math

class_num = 10
batch_size = 4
label = torch.LongTensor(batch_size, 1).random_() % class_num
print(label)
# tensor([[6],
#        [0],
#        [3],
#        [2]])
t = torch.zeros(batch_size, class_num).scatter_(1, label, torch.ones((4,1)))
print(t)
print(13)

print(13 * 4)
# print(13412)
