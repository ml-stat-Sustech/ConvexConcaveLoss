import abc
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MembershipInferenceAttack(abc.ABC):
    """
    Abstract base class for membership inference attack classes.
    """

    def __init__(self, ):

        super().__init__()

    @staticmethod
    def cal_metrics(label, pred_label, pred_posteriors):
        num_nans = np.sum(np.isnan(pred_posteriors))
        print(f"Number of NaN values in pred_posteriors: {num_nans}")
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, zero_division=0)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)

        return acc, precision, recall, f1, auc

    @staticmethod
    def cal_metric_for_class(self, label, pred_label, pred_posteriors, original_target_labels):
        """
        Calculate metrics for each class of the train (shadow) or test (target) dataset
        """

        class_list = sorted(list(set(original_target_labels)))
        for class_idx in class_list:
            subset_label = []
            subset_pred_label = []
            subset_pred_posteriors = []
            for i in range(len(label)):
                if original_target_labels[i] == class_idx:
                    subset_label.append(label[i])
                    subset_pred_label.append(pred_label[i])
                    subset_pred_posteriors.append(pred_posteriors[i])
                    # only contain subset
            if len(subset_label) != 0:
                acc, precision, recall, f1, auc = self.cal_metrics(
                    subset_label, subset_pred_label, subset_pred_posteriors)

                print('Acc for class %d: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f' %
                      (class_idx, 100. * acc, precision, recall, f1, auc))
