import torch
from torch import nn
from torch.nn import functional as F

from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .hiarachical_layers import bn_selector, downsample_selector
from .ma_learning import GraphConvolution, gen_A, gen_adj
from .Basenet import Basenet
from .feature_learning import BasicBlock, ResNetbasic, BasicBlock2, ResNetbasic2