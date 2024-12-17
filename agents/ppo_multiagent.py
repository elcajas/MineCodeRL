import numpy as np
import time
import os
import logging
from gymnasium import error

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from mineclip import MineCLIP
from mineclip import SimpleFeatureFusion
from mineclip.mineagent.batch import Batch
from mineclip.mineagent.actor.distribution import MultiCategorical
from mineclip.utils import build_mlp

from transformers import AutoProcessor

from .utils import set_MineCLIP, set_gDINO, layer_init, set_hf_gDINO
from .inference import predict
from agents import features_mlp as F
from .encoders import ImageEncoder