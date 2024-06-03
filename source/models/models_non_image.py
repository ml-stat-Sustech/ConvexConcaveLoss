import torch
import torch.nn as nn


class TexasClassifier(nn.Module):
    def __init__(self, num_classes=100, droprate=None):
        super(TexasClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6169, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        if droprate is not None:
            self.classifier = nn.Sequential(nn.Dropout(droprate),
                                            nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


class PurchaseClassifier(nn.Module):
    def __init__(self, num_classes=100, droprate=None):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        if droprate is not None:
            self.classifier = nn.Sequential(nn.Dropout(droprate),
                                            nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


def Purchase(**kwargs):
    return PurchaseClassifier(**kwargs)


def Texas(**kwargs):
    return TexasClassifier(**kwargs)
