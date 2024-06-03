import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from attacks.membership_inference.membership_Inference_attack import MembershipInferenceAttack
from models.attack_model import MLP_BLACKBOX
from utility.main_parse import save_dict_to_yaml
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class AttackDataset(Dataset):
    def __init__(self, data, labels):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class AugemtaionAttackDataset(Dataset):
    def __init__(self, args, attack_type , target_model, shadow_model,
                                        target_train_dataset, target_test_dataset, shadow_train_dataset, shadow_test_dataset,device):
        
        
        train_object = AugmentedAttackData(shadow_model,shadow_train_dataset, shadow_test_dataset, device =device, batch_size =args.batch_size, augment_kwarg_translation = args.augment_kwarg_translation, augment_kwarg_rotation =args.augment_kwarg_rotation)
        test_object = AugmentedAttackData(target_model,target_train_dataset, target_test_dataset,device =device, batch_size =args.batch_size, augment_kwarg_translation = args.augment_kwarg_translation, augment_kwarg_rotation =args.augment_kwarg_rotation)
        
        
        
        train_dataset, train_label = train_object.augmentation_attack(attack_type)
        print(f"dataset shape: {train_dataset.shape}, label shape:  {train_label.shape}")
        test_dataset, test_label =  test_object.augmentation_attack(attack_type)

        self.attack_train_dataset  =AttackDataset(train_dataset,train_label)
        self.attack_test_dataset = AttackDataset(test_dataset, test_label)

class AugmentedAttackData(object):
    """
    Dataset that applies augmentation and can perform augmentation attack.
    """

    def __init__(self,model, train_dataset, test_dataset , device,batch_size = 128, augment_kwarg_translation=1,augment_kwarg_rotation =1, max_samples=100):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.device = device
        #self.attack_type = attack_type
        self.augment_kwarg_rotation = augment_kwarg_rotation
        self.augment_kwarg_translation = augment_kwarg_translation
        self.max_samples = max_samples
        self.create_augments()
        #self.attack_set = self.augmentation_attack()
        self.batch_size = batch_size

    def create_rotation_augments(r):
        angles = [-r, 0, r]  
        return angles
    
    def create_translation_augments(self, d):
        """
        Create all possible combinations of translations that satisfy |i| + |j| = d
        """
        combinations = []
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                if abs(i) + abs(j) == d:
                    combinations.append((i, j))
        combinations.append((0,0))            
        return combinations
    
    def create_augments(self):
        
        self.augments_rotation = [-self.augment_kwarg_rotation,0 ,self.augment_kwarg_rotation]
        self.augments_translation = self.create_translation_augments(self.augment_kwarg_translation)

    

    
    def _get_transform(self, augment, attack_type):
        if attack_type == 'rotation':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda img: TF.rotate(img, angle=augment)),
                transforms.ToTensor()
            ])
        elif attack_type == 'translation':
            # Define translation transform here

            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda img: TF.affine(img, angle=0, translate=augment, scale=1, shear=0)),
                transforms.ToTensor()
            ])
            
        else:
            return None

    def check_correct(self, predictions, labels):
        return labels.eq(predictions.argmax(dim=1)).float()



 
    def augmentation_attack(self, attack_type):
        attack_results = []

        if attack_type =="rotation":
            augments = self.augments_rotation
        elif attack_type == "translation":
            augments = self.augments_translation
        else: raise ValueError("Not rotation or translation")
        
        
        with torch.no_grad():
            self.model.eval()
            for i, augment in enumerate(augments):
                transform = self._get_transform(augment, attack_type)
                train_augmented_dataset = CustomDataset(self.train_dataset, transform)
                train_data_loader = DataLoader(train_augmented_dataset, batch_size=self.batch_size, shuffle=False)
                test_augmented_dataset = CustomDataset(self.test_dataset, transform)
                test_data_loader = DataLoader(test_augmented_dataset, batch_size=self.batch_size, shuffle=False)
                
                batch_results  = []
                for batch, labels in train_data_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(batch)
                    correct = self.check_correct(outputs, labels)
                    batch_results.append(correct)
                
                for batch, labels in test_data_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(batch)
                    correct = self.check_correct(outputs, labels)
                    batch_results.append(correct)
                
                
                attack_results.append(torch.cat(batch_results).cpu().numpy())

      
   
        final_results = np.stack(attack_results, axis=1)
   
        m = np.concatenate([np.ones(len(self.train_dataset)), np.zeros(len(self.test_dataset))], axis=0)
        print("dataset is finished")
        
        return final_results, m

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class DataAugmentationMIA(MembershipInferenceAttack):
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
        #if self.attack_type == "data_augmentation": 
        self.attack_model = MLP_BLACKBOX(dim_in=self.num_class).to(self.device)
        #raise ValueError("Not implemented yet")

        #self.attack_model = self.attack_model
        self.criterion = nn.CrossEntropyLoss()
        self.train(self.attack_train_loader)

    def train(self, dataloader, train_epoch=5):
        self.attack_model.train()
        self.optimizer = torch.optim.Adam(
            self.attack_model.parameters(), lr=0.001)

        for e in range(1, train_epoch + 1):
            train_loss = 0
            #print(torch.get_default_dtype())  # Check default data type
            #torch.set_default_dtype(torch.float)  # Set default to float if needed
            labels = []
            pred_labels = []
            pred_posteriors = []
            for inputs, targets in dataloader:
                #print(f"input: {inputs.shape}, target: {targets.shape}")
                #print(inputs)
                #print(targets) 
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(
                    self.device).float(), targets.to(self.device).long()
                outputs = self.attack_model(inputs)
                #print(f"output: {outputs.shape}")
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
            print(f'Epoch: {e}, Overall Train Acc: {train_acc*100:.3f}%, precision: {train_precision:.3f}, recall: {train_recall:.3f}, f1: {train_f1:.3f}, auc: {train_auc:.3f}')
            
            print(f'Epoch: {e}, Overall Test Acc: {test_acc*100:.3f}%, precision: {test_precision:.3f}, recall: {test_recall:.3f}, f1: {test_f1:.3f}, auc: {test_auc:.3f}\n\n')

            train_tuple = (train_acc, train_precision,
                           train_recall, train_f1, train_auc)
            test_tuple = (test_acc, test_precision,
                          test_recall, test_f1, test_auc)

            if e == train_epoch:
                mia_bb_dict ={f'mia_data_augmentation_{self.attack_type}_epoch': e, 
                              f"data_augmentation_{self.attack_type} train_acc": train_acc, 
                              f"data_augmentation_{self.attack_type} train_precision" : train_precision, 
                           f"data_augmentation_{self.attack_type} recall" : train_recall, 
                           f"data_augmentation_{self.attack_type} train_f1" : train_f1, 
                           f"data_augmentation_{self.attack_type} train_auc" :train_auc , 
                           f"data_augmentation_{self.attack_type} test_acc" : test_acc , 
                           f"data_augmentation{self.attack_type} test_precision" : test_precision,  
                           f"data_augmentation_{self.attack_type} test_recall" : test_recall, 
                           f"data_augmentation_{self.attack_type} test_f1" :test_f1 ,
                           f"data_augmentation_{self.attack_type} test_auc" :test_auc}
                new_dict = {}
                for key, value_tuple in mia_bb_dict.items():
                    new_dict[key] = float(value_tuple)
                
                save_dict_to_yaml(new_dict, f'{self.save_path}/mia_data_augmentation_{self.attack_type}.yaml')
            
        return train_tuple, test_tuple, test_results

    def infer(self, dataloader):
        self.attack_model.eval()
        original_target_labels = []
        labels = []
        pred_labels = []
        pred_posteriors = []
        with torch.no_grad():
            for inputs, targets in dataloader:

                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()
                pred_posteriors += posteriors.cpu().tolist()


            pred_posteriors = [row[1] for row in pred_posteriors]

            test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
                labels, pred_labels, pred_posteriors)
            

            test_results = {"test_mem_label": labels,
                            "test_pred_label": pred_labels,
                            "test_pred_prob": pred_posteriors,}

        self.attack_model.train()
        return test_acc, test_precision, test_recall, test_f1, test_auc, test_results