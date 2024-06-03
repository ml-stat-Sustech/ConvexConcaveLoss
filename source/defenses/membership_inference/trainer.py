
import abc
from models.attack_model import MLP_BLACKBOX
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from runx.logx import logx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class Trainer(abc.ABC):
    """
    Abstract base class for MIA defenses.
    """

    def __init__(self,):

        super().__init__()

    @abc.abstractmethod
    def train(self, dataloader, train_epoch, **kwargs):
        raise NotImplementedError

    # @abc.abstractmethod
    # def infer(self, datalaoder, **kwargs):
    #     raise NotImplementedError
