

import torch
import torch.nn as nn
import torch.nn.functional as F


class MembershipInferenceAttackModel(nn.Module):
    """
    Implementation of a pytorch model for learning a membership inference attack.

    The features used are probabilities/logits or losses for the attack training data along with
    its true labels.
    """

    def __init__(self, num_classes, num_features=None):

        self.num_classes = num_classes
        if num_features:
            self.num_features = num_features
        else:
            self.num_features = num_classes

        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(self.num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(64 * 2, 2),
        )

        # self.output = nn.Sigmoid()

    def forward(self, mi_feature, label):
        """Forward the model."""
        out_x = self.features(mi_feature)
        out_l = self.labels(label)
        pred = self.combine(torch.cat((out_x, out_l), 1))

        return pred
