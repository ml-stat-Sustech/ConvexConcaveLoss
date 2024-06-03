import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from scipy.stats import kurtosis, skew
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np
from attacks.membership_inference.attack_utils import phi_stable_batch_epsilon, m_entr_comp


def compute_norm_metrics(gradient):
    """Compute the metrics"""
    l1 = np.linalg.norm(gradient, ord=1)
    l2 = np.linalg.norm(gradient)
    min = np.linalg.norm(gradient, ord=-np.inf)  ## min(abs(x))
    max = np.linalg.norm(gradient, ord=np.inf)  ## max(abs(x))
    mean = np.average(gradient)
    skewness = skew(gradient)
    kurtosis_val = kurtosis(gradient)
    return [l1, l2, min, max, mean, skewness, kurtosis_val]

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features


class ModelParser:
    """
    ModelParser handles what information should be extracted from the target/shadow model
    """

    def __init__(self, args, model):
        self.target_list = None
        self.posteriors = None
        self.args = args
        self.device = self.args.device
        self.model = model.to(self.device)
        self.model.eval()
        # self.criterion = get_loss(loss_type= args.loss_type, device = args.device, args= args, num_classes=args.num_class)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    def combined_gradient_attack(self, dataloader):
        """Gradient attack w.r.t input and weights"""
        self.model.eval()
        target_list = []
        # store results
        names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
        all_stats_x = {name: [] for name in names}
        all_stats_w = {name: [] for name in names}

        # iterate over batches
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True  # Enable gradient computation w.r.t inputs

            # Compute output and loss
            outputs = self.model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)

            target_list += targets.cpu().tolist()

            # Zero gradients, perform a backward pass, and get the gradients
            self.model.zero_grad()
            loss.backward()

            # Gradients w.r.t input
            gradients_x = inputs.grad.view(inputs.size(0), -1).cpu().numpy()

            # Gradients w.r.t weights
            grads_onesample = []
            for param in self.model.parameters():
                grads_onesample.append(param.grad.view(-1))
            gradient_w = torch.cat(grads_onesample).cpu().numpy()

            # Compute and store statistics for each sample in the batch
            for gradient in gradients_x:
                stats = compute_norm_metrics(gradient)
                for i, stat in enumerate(stats):
                    all_stats_x[names[i]].append(stat)

            # Assuming the gradients w.r.t weights are the same for all samples in the batch
            stats = compute_norm_metrics(gradient_w)
            for i, stat in enumerate(stats):
                all_stats_w[names[i]].extend([stat] * len(inputs))

        # Convert lists to numpy arrays
        for name in names:
            all_stats_x[name] = np.array(all_stats_x[name])
            all_stats_w[name] = np.array(all_stats_w[name])

        return {"targets" :target_list, "gird_x_w": (all_stats_x, all_stats_w)}
    def get_posteriors(self, dataloader):
        target_list = []
        posteriors_list = []
        with torch.no_grad():
            for btch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                # print(posteriors.shape) torch.Size([128, 10])

                # add loss
                # losses = self.criterion(outputs, targets)
                # print(losses)
                # exit()
                target_list += targets.cpu().tolist()
                posteriors_list += posteriors.detach().cpu().numpy().tolist()
                # all_losses += losses.tolist()
        torch.cuda.empty_cache()
        self.posteriors = posteriors_list
        self.target_list = target_list
        return {"targets": target_list, "posteriors": posteriors_list}
        # targets :1-10

    def get_metrics(self, dataloader):
        target_list = []
        posteriors_list = []
        losses_list = []
        entropies_list = []
        confidences_list = []
        correctness_list = []
        phi_stable_list = []
        modified_entropy_list = []
        #print("calculate metrics!")
        with torch.no_grad():
            #breakpoint()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                posteriors = F.softmax(outputs, dim=1).cpu().numpy()

                loss = self.criterion(outputs, targets).cpu().numpy()
                losses_list.extend(loss)

                entropy = -np.sum(posteriors * np.log(posteriors + 1e-10), axis=1)
                entropies_list.extend(entropy)

                confidence, _ = torch.max(F.softmax(outputs, dim=1), 1)
                confidences_list.extend(confidence.cpu().numpy())

                correct = outputs.argmax(dim=1) == targets
                correctness_list.extend(correct.cpu().numpy())

                phi_stable = phi_stable_batch_epsilon(posteriors, targets.cpu().numpy())
                phi_stable_list.extend(phi_stable)

                modified_entropy = m_entr_comp(posteriors, targets.cpu().numpy())
                modified_entropy_list.extend(modified_entropy)

                target_list.extend(targets.cpu().numpy())
                posteriors_list.extend(posteriors)

        # Convert lists to NumPy arrays

        
        self.targets = np.array(target_list)
        self.posteriors = posteriors_list
        self.losses = np.array(losses_list)
        self.entropies = np.array(entropies_list)
        self.confidences = np.array(confidences_list)
        self.correctness = np.array(correctness_list).astype(int)
        self.phi_stable = np.array(phi_stable_list)
        self.modified_entropy = np.array(modified_entropy_list)
        # print(type(self.losses))
        # print(self.losses)
        
        
        
        return {
            "targets": self.targets,
            "posteriors": self.posteriors,
            "losses": self.losses,
            "entropies": self.entropies,
            "confidences": self.confidences,
            "correctness": self.correctness,
            "phi_stable": self.phi_stable,
            "modified_entropies":self.modified_entropy
        }

    def parse_info_whitebox(self, dataloader, layers):
        target_list = []
        posteriors_list = []
        embedding_list = []
        loss_list = []
        self.individual_criterion = nn.CrossEntropyLoss(reduction='none')
        self.model_feature = FeatureExtractor(self.model, layers=layers)

        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)  # can be reduced
            features = self.model_feature(inputs)

            emb = features[layers[-2]]  # can further specified
            emb = torch.flatten(emb, start_dim=1).detach().cpu().tolist()

            losses = self.individual_criterion(outputs, targets)

            losses = losses.detach().cpu().tolist()
            posteriors = F.softmax(outputs, dim=1).detach().cpu().tolist()

            target_list += targets.cpu().tolist()
            embedding_list += emb
            posteriors_list += posteriors
            loss_list += losses
        info = {"targets": target_list, "embeddings": embedding_list,
                "posteriors": posteriors_list, "losses": loss_list}
        return info


class AttackDataset:
    """
    Generate attack dataset
    """

    def __init__(self, args, attack_type, target_model, shadow_model, target_train_dataloader, target_test_dataloader,
                 shadow_train_dataloader, shadow_test_dataloader):
        self.args = args
        self.attack_type = attack_type
        self.target_model_parser = ModelParser(args, target_model)
        self.shadow_model_parser = ModelParser(args, shadow_model)

        if attack_type == "white_box":
            self.target_train_info = self.target_model_parser.combined_gradient_attack(target_train_dataloader)
            self.target_test_info = self.target_model_parser.combined_gradient_attack(target_test_dataloader)
            self.shadow_train_info = self.shadow_model_parser.combined_gradient_attack(shadow_train_dataloader)
            self.shadow_test_info = self.shadow_model_parser.combined_gradient_attack(shadow_test_dataloader)


        else:
            # if attack_type == "black-box":
            self.target_train_info = self.target_model_parser.get_posteriors(
                target_train_dataloader)
            self.target_test_info = self.target_model_parser.get_posteriors(
                target_test_dataloader)
            self.shadow_train_info = self.shadow_model_parser.get_posteriors(
                shadow_train_dataloader)
            self.shadow_test_info = self.shadow_model_parser.get_posteriors(
                shadow_test_dataloader)
            # print(self.target_train_info)
            # exit()
            # _info contains posteriors, it is prob
            # get_posteriors : return {"targets": target_list, "posteriors": posteriors_list}
            # get attack dataset
        self.attack_train_dataset, self.attack_test_dataset = self.generate_attack_dataset()

    def parse_info(self, info, label=0):
        mem_label = [label] * len(info["targets"])
        original_label = info["targets"]
        parse_type = self.attack_type
        if parse_type in ["black-box","black_box"]:
            mem_data = info["posteriors"]
        elif parse_type == "black-box-sorted":
            mem_data = [sorted(row, reverse=True)
                        for row in info["posteriors"]]
        elif parse_type == "black-box-top3":
            mem_data = [sorted(row, reverse=True)[:3]
                        for row in info["posteriors"]]
        elif parse_type == "metric-based":
            mem_data = info["posteriors"]
        elif parse_type == "white_box":
            mem_data = info["gird_x_w"]

        else:
            raise ValueError("More implementation is needed :P")
        return mem_data, mem_label, original_label

        ## mem_data posteriors prob; mem_label 0 or 1 ; original_label : class label 1-10

    def generate_attack_dataset(self):

        mem_data0, mem_label0, original_label0 = self.parse_info(
            self.target_train_info, label=1)
        mem_data1, mem_label1, original_label1 = self.parse_info(
            self.target_test_info, label=0)
        mem_data2, mem_label2, original_label2 = self.parse_info(
            self.shadow_train_info, label=1)
        mem_data3, mem_label3, original_label3 = self.parse_info(
            self.shadow_test_info, label=0)

        ## shadow_train_info and shadow_test_info construct the attack_train_dataset, means that using shadow model to train classification model
        if self.attack_type == "white_box":
            attack_train_dataset = [
                {"shadow_train_data": mem_data0, "shadow_test_data": mem_data1},
                {"shadow_train_mem_label": mem_label0, "shadow_test_mem_label": mem_label1},
                {"shadow_train_label": original_label0, "shadow_test_label": original_label1}
            ]
        else:
            attack_train_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(np.array(mem_data2 + mem_data3, dtype='f')),
                torch.from_numpy(np.array(mem_label2 + mem_label3)
                                 ).type(torch.long),
                torch.from_numpy(np.array(original_label2 +
                                          original_label3)).type(torch.long),
            )

        # attack_train_dataset
        # (tensor([2.4628e-04, 3.9297e-01, 3.7773e-04, 9.3001e-05, 2.3009e-04, 3.6489e-04, 1.2535e-04, 6.5634e-05, 1.3375e-03, 6.0419e-01]), tensor(1), tensor(9))

        if self.attack_type == "white_box":
            attack_test_dataset = [
                {"target_train_data": mem_data0, "target_test_data": mem_data1},
                {"target_train_mem_label": mem_label0, "target_test_mem_label": mem_label1},
                {"target_train_label": original_label0, "target_test_label": original_label1}
            ]
        else:
            attack_test_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(np.array(mem_data0 + mem_data1, dtype='f')),
                torch.from_numpy(np.array(mem_label0 + mem_label1)
                                 ).type(torch.long),
                torch.from_numpy(np.array(original_label0 +
                                          original_label1)).type(torch.long),
            )

        return attack_train_dataset, attack_test_dataset