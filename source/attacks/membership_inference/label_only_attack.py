
import sys

from attacks.membership_inference.membership_Inference_attack import MembershipInferenceAttack

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from source.utility.main_parse import save_dict_to_yaml
# from source.models.utils import FeatureExtractor, VerboseExecution
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
class LabelOnlyMIA(MembershipInferenceAttack):
    def __init__(
            self,
            device,
            target_model,
            shadow_model,
            save_path,
            target_loader=None,
            shadow_loader=None,
            input_shape=(3, 32, 32),
            nb_classes=10,
            batch_size=1000,
    ):

        super().__init__()

        self.device = device
        self.target_train_dataset = target_loader[0].dataset
        self.target_test_dataset = target_loader[1].dataset
        self.target_train_loader = torch.utils.data.DataLoader(
            self.target_train_dataset, batch_size=batch_size, shuffle=True)
        self.target_test_loader = torch.utils.data.DataLoader(
            self.target_test_dataset, batch_size=batch_size, shuffle=True)
        self.target_model = target_model

        self.shadow_train_dataset = shadow_loader[0].dataset
        self.shadow_test_dataset = shadow_loader[1].dataset
        self.shadow_train_loader = torch.utils.data.DataLoader(
            self.shadow_train_dataset, batch_size=batch_size, shuffle=True)
        self.shadow_test_loader = torch.utils.data.DataLoader(
            self.shadow_test_dataset, batch_size=batch_size, shuffle=True)
        self.shadow_model = shadow_model

        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.save_path = save_path

    def SearchThreshold(self):
        ARTclassifier = PyTorchClassifier(
            model=self.shadow_model,
            clip_values=None,
            loss=F.cross_entropy,
            input_shape=self.input_shape,
            nb_classes=self.nb_classes
        )

        for idx, (data, label) in enumerate(self.shadow_train_loader):
            x_train = data.numpy() if idx == 0 else np.concatenate((x_train, data.numpy()), axis=0)
            y_train = label.numpy() if idx == 0 else np.concatenate((y_train, label.numpy()), axis=0)

        for idx, (data, label) in enumerate(self.shadow_test_loader):
            x_test = data.numpy() if idx == 0 else np.concatenate((x_test, data.numpy()), axis=0)
            y_test = label.numpy() if idx == 0 else np.concatenate((y_test, label.numpy()), axis=0)

        Attack = LabelOnlyDecisionBoundary(estimator=ARTclassifier)
        Attack.calibrate_distance_threshold(x_train, y_train, x_test, y_test)
        distance_threshold = Attack.distance_threshold_tau
        return distance_threshold

    def Infer(self):

        thd = self.SearchThreshold()

        ARTclassifier = PyTorchClassifier(
            model=self.target_model,
            clip_values=None,
            loss=F.cross_entropy,
            input_shape=self.input_shape,
            nb_classes=self.nb_classes
        )

        for idx, (data, label) in enumerate(self.target_train_loader):
            x_train = data.numpy() if idx == 0 else np.concatenate((x_train, data.numpy()), axis=0)
            y_train = label.numpy() if idx == 0 else np.concatenate((y_train, label.numpy()), axis=0)

        for idx, (data, label) in enumerate(self.target_test_loader):
            x_test = data.numpy() if idx == 0 else np.concatenate((x_test, data.numpy()), axis=0)
            y_test = label.numpy() if idx == 0 else np.concatenate((y_test, label.numpy()), axis=0)

        Attack = LabelOnlyDecisionBoundary(estimator=ARTclassifier, distance_threshold_tau=thd)

        x_train_test = np.concatenate((x_train, x_test), axis=0)
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        prediction = Attack.infer(x_train_test, y_train_test)

        member_groundtruth = np.ones(int(len(x_train)))
        non_member_groundtruth = np.zeros(int(len(x_train)))
        groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))

        binary_predictions = (prediction > 0.5).astype(int)

        recall = recall_score(groundtruth, binary_predictions)
        precision = precision_score(groundtruth, binary_predictions)
        f1 = f1_score(groundtruth, binary_predictions)
        acc = accuracy_score(groundtruth, binary_predictions)
        auc = roc_auc_score(groundtruth, prediction)
        label_only_dict = {"label_only_acc": acc, "label_only_recall": recall, "label_only_acc_f1": f1,
                           "label_only_precision": precision, "label_only_auc": auc}

        save_dict_to_yaml(label_only_dict, f"{self.save_path}/label_only_attacks.yaml")

        fpr, tpr, _ = roc_curve(groundtruth, prediction, pos_label=1, drop_intermediate=False)
        AUC = round(auc(fpr, tpr), 4)
        return AUC