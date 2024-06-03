import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from source.attacks.membership_inference.membership_Inference_attack import MembershipInferenceAttack
from source.utility.main_parse import save_dict_to_yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from source.models.attack_model import MLP_BLACKBOX

class BlackBoxMIA(MembershipInferenceAttack):
    def __init__(
            self,
            num_class,
            device,
            attack_type,
            attack_train_dataset,
            attack_test_dataset,
            save_path,
            batch_size=128):

        super().__init__()

        self.num_class = num_class
        self.device = device
        self.attack_type = attack_type
        self.attack_train_dataset = attack_train_dataset
        self.attack_test_dataset = attack_test_dataset
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train_dataset, batch_size=batch_size, shuffle=True)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test_dataset, batch_size=batch_size, shuffle=False)
        self.save_path = save_path
        if self.attack_type in ["black-box","black_box"]:
            self.attack_model = MLP_BLACKBOX(dim_in=self.num_class)
        elif self.attack_type == "black-box-sorted":
            self.attack_model = MLP_BLACKBOX(dim_in=self.num_class)
        elif self.attack_type == "black-box-top3":
            self.attack_model = MLP_BLACKBOX(dim_in=3)
        else:
            raise ValueError("Not implemented yet")

        self.attack_model = self.attack_model.to(self.device).float()
        self.criterion = nn.CrossEntropyLoss()
        self.train(self.attack_train_loader)

    def train(self, dataloader, train_epoch=100):
        #print(torch.get_default_dtype())   Check default data type
        torch.set_default_dtype(torch.float)  # Set default to float if needed
        self.attack_model.train()
        self.optimizer = torch.optim.Adam(
            self.attack_model.parameters(), lr=0.001)
        for e in range(1, train_epoch + 1):
            train_loss = 0
            labels = []
            pred_labels = []
            pred_posteriors = []
            for inputs, targets, original_labels in dataloader:
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()
                pred_posteriors += posteriors.cpu().tolist()

            pred_posteriors = [row[1] for row in pred_posteriors]

            train_acc, train_precision, train_recall, train_f1, train_auc = super().cal_metrics(
                labels, pred_labels, pred_posteriors)
            test_acc, test_precision, test_recall, test_f1, test_auc, test_results = self.infer(
                self.attack_test_loader)
            print('Epoch: %d, Overall Train Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f' % (
                e, 100. * train_acc, train_precision, train_recall, train_f1, train_auc))
            print('Epoch: %d, Overall Test Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f\n\n' % (
                e, 100. * test_acc, test_precision, test_recall, test_f1, test_auc))

            train_tuple = (train_acc, train_precision,
                           train_recall, train_f1, train_auc)
            test_tuple = (test_acc, test_precision,
                          test_recall, test_f1, test_auc)

            if e == train_epoch:
                mia_bb_dict = {'mia_black_box_epoch': e, "black-box train_acc": train_acc,
                               "black-box train_precision": train_precision,
                               "black-box recall": train_recall, "black-box train_f1": train_f1,
                               "black-box train_auc": train_auc,
                               "black-box test_acc": test_acc, "black-box test_precision": test_precision,
                               "black-box test_recall": test_recall, "black-box test_f1": test_f1,
                               "black-box test_auc": test_auc}
                new_dict = {}
                for key, value_tuple in mia_bb_dict.items():
                    new_dict[key] = float(value_tuple)

                save_dict_to_yaml(new_dict, f'{self.save_path}/mia_black_box.yaml')

        return train_tuple, test_tuple, test_results

    def infer(self, dataloader):
        self.attack_model.eval()
        original_target_labels = []
        labels = []
        pred_labels = []
        pred_posteriors = []
        with torch.no_grad():
            for inputs, targets, original_labels in dataloader:
                inputs, targets, original_labels = inputs.to(self.device), targets.to(
                    self.device), original_labels.to(self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()
                pred_posteriors += posteriors.cpu().tolist()
                original_target_labels += original_labels.cpu().tolist()

            pred_posteriors = [row[1] for row in pred_posteriors]

            test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
                labels, pred_labels, pred_posteriors)
            # print single class performance
            super().cal_metric_for_class(super(),
                                         labels, pred_labels, pred_posteriors, original_target_labels)

            test_results = {"test_mem_label": labels,
                            "test_pred_label": pred_labels,
                            "test_pred_prob": pred_posteriors,
                            "test_target_label": original_target_labels}

        self.attack_model.train()
        return test_acc, test_precision, test_recall, test_f1, test_auc, test_results
