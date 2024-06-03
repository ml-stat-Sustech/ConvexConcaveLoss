import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import pandas as pd
def merge_dict_of_np_arrays(dicts):
    merged = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            merged[key].append(value)
    
    for key, value_list in merged.items():
        merged[key] = np.concatenate(value_list)

    return dict(merged)



class Metrics:
    def __init__(self, labels, logits=None, probs=None):
        if probs is not None:
            self.probs = probs
        elif logits is not None:
            self.probs = F.softmax(logits, dim=1)
        else:
            raise ValueError("Both logits and probs cannot be None.")

        self.labels = labels
        self.entropy = self._entropy()
        self.m_entropy = self._m_entropy()
        self.loss = F.cross_entropy(logits, labels, reduction="none") if logits is not None else None
        self.py = torch.exp(-self.loss)
        self.confidence, _ = torch.max(self.probs, dim=1)
        self.phi_stable = self.phi_stable_batch_epsilon()
        
        self.metrics = {
            "loss": self.loss.cpu().numpy(),
            "entropy" : self.entropy.cpu().numpy(),
            "m_entropy": self.m_entropy.cpu().numpy(),
            "py": self.py.cpu().numpy(),
            "confidence": self.confidence.cpu().numpy(),
            "phi_stable":self.phi_stable.cpu().numpy()
        }
    def _log_value(self, probs, small_value=1e-30):
        return -torch.log(torch.maximum(probs, torch.tensor(small_value)))

    def _entropy(self):
        return torch.sum(torch.mul(self.probs, self._log_value(self.probs)), dim=1)

    def _m_entropy(self):
        log_probs = self._log_value(self.probs)
        reverse_probs = 1 - self.probs
        log_reverse_probs = self._log_value(reverse_probs)

        modified_probs = self.probs.clone()
        modified_probs[torch.arange(self.labels.size(0)), self.labels] = reverse_probs[torch.arange(self.labels.size(0)), self.labels]

        modified_log_probs = log_reverse_probs.clone()
        modified_log_probs[torch.arange(self.labels.size(0)), self.labels] = log_probs[torch.arange(self.labels.size(0)), self.labels]

        return torch.sum(torch.mul(modified_probs, modified_log_probs), dim=1)

    def phi_stable_batch_epsilon(self, epsilon=1e-10):
        posterior_probs = self.probs + epsilon

        one_hot_labels = torch.zeros_like(posterior_probs)
        one_hot_labels[torch.arange(self.labels.size(0)), self.labels] = 1

        log_likelihood_correct = torch.log(posterior_probs[torch.arange(self.labels.size(0)), self.labels])
        sum_incorrect = torch.sum(posterior_probs * (1 - one_hot_labels), dim=1)
        sum_incorrect = torch.clamp(sum_incorrect, min=epsilon)

        log_likelihood_incorrect = torch.log(sum_incorrect)
        phi_stable = log_likelihood_correct - log_likelihood_incorrect

        return phi_stable

class StaMetrics:
    def __init__(self):
        self._metrics_list = []
        self.sta_epochs = pd.DataFrame()

    def add_metrics(self, metrics):
        self._metrics_list.append(metrics)

    def add_total_variance(self, epoch, loader_type):
        sta_book = {}

        metric_book = merge_dict_of_np_arrays(self._metrics_list)
        for key, value_np in metric_book.items():
            sta_book[f"{key}_std"] = np.std(value_np)
            sta_book[f"{key}_mean"] = np.mean(value_np)
            sta_book[f"{key}_var"] = np.var(value_np)
        sta_book["epoch"] = epoch
        sta_book["loader_type"] = loader_type
        self.sta_epochs = pd.concat([self.sta_epochs,pd.DataFrame([sta_book])], ignore_index=True)
        self._metrics_list = []



# class StaMetrics:
#     def __init__(self):
        
#         self._metrics_list = []
#         #self.sta_book = {}
#         self.sta_epochs = pd.DataFrame()
        

#     def add_metrics(self, metrics):
#         self._metrics_list.append(metrics)
    
#     def add_total_variance(self, epoch, loader_type):
#         sta_book = {}
        
        
#         metric_book = merge_dict_of_np_arrays(self._metrics_list)
#         for key, value_np in metric_book.items():
#             sta_book[f"{key}_std"] = np.std(value_np)
#             sta_book[f"{key}_mean"] = np.mean(value_np)
#             sta_book[f"{key}_var"] = np.var(value_np)
#         sta_book["epoch"] = epoch
#         sta_book["loader_type"] = loader_type
#         self._metrics_list = []
#         self.sta_epochs.append(sta_book, ignore_index=True)
    
        
