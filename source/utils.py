
import sys
import numpy as np
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from torch.optim.adamw import AdamW

from models.models_non_image import  Purchase,Texas
import yaml
from models.resnet import resnet20
import torch.nn as nn
import torchvision
#from transformers import ViTForImageClassification
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch, gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, Dataset
#from transformers import AdamW
sys.path.append("..")
sys.path.append("../..")
def store_dict_to_yaml(my_dict, save_path, file_name):
    """
    Stores a dictionary into a YAML file at the specified file path.
    
    Parameters:
    my_dict (dict): The dictionary to be stored.
    file_path (str): The path where the YAML file should be created.
    
    Returns:
    None
    """
    for key, value in my_dict.items():
        if isinstance(value, (np.ndarray, np.generic)):
            my_dict[key] = value.item() 
    file_path = f'{save_path}/{file_name}'
    with open(file_path, 'w') as file:
        yaml.dump(my_dict, file)

import importlib

def p1_score(acc, adv):
    return 2*acc*(1-adv)/(acc+1-adv)

def call_function_from_module(module_name, function_name):
    """
    Dynamically import a module and call a specified function within that module.

    Args:
    - module_name (str): The name of the module to import.
    - function_name (str): The name of the function to call within the module.

    Returns:
    - The return value of the called function.

    Raises:
    - ImportError: If the module cannot be imported.
    - AttributeError: If the function is not found in the module.
    - Exception: If an error occurs during the function call.
    """

    try:
        # Dynamically import the specified module
        mod = importlib.import_module(module_name)

        # Get the specified function from the module
        func = getattr(mod, function_name)

        # Call the function and return its result
        return func
    except ImportError as e:
        print(f"Error importing module: {e}")
        raise
    except AttributeError as e:
        print(f"Function not found in the module: {e}")
        raise
    except Exception as e:
        print(f"Error calling the function: {e}")
        raise

def dict_str(input_dict):
    for key, value in input_dict.items():
        input_dict[key] = str(value)
    return input_dict
def get_init_args(obj):
    init_args = {}
    for attr_name in dir(obj):
        if not callable(getattr(obj, attr_name)) and not attr_name.startswith("__"):
            init_args[attr_name] = getattr(obj, attr_name)
    return init_args


def generate_save_path_1(opt):
    if isinstance(opt, dict):
        opt = argparse.Namespace(**opt)
    save_path1 = f'{opt.log_path}/{opt.dataset}/{opt.model}/{opt.training_type}'
    return save_path1

def generate_save_path_2(opt):
    if isinstance(opt, dict):
        opt = argparse.Namespace(**opt)
    #temp_save = str(opt.temp).rstrip('0').rstrip('.') if '.' in str(opt.temp) else str(opt.temp)
    temp_save= standard_float(opt.temp)
    alpha_save = standard_float(opt.alpha)
    gamma_save = standard_float(opt.gamma)
    tau_save = standard_float(opt.tau)
    #alpha_save = str(opt.alpha).rstrip('0').rstrip('.') if '.' in str(opt.alpha) else str(opt.alpha)
    save_path2 =  f"{opt.loss_type}/epochs{opt.epochs}/seed{opt.seed}/{temp_save}/{alpha_save}/{gamma_save}/{tau_save}"
    return save_path2
        
def standard_float(hyper_parameter):
    return str(hyper_parameter).rstrip('0').rstrip('.') if '.' in str(hyper_parameter) else str(hyper_parameter)
    


def generate_save_path(opt, mode = None):
    if isinstance(opt, dict):
        opt = argparse.Namespace(**opt)
    if mode == None:
        save_pth = f'{generate_save_path_1(opt)}/{opt.mode}/{generate_save_path_2(opt)}'
    else:
        save_pth = f'{generate_save_path_1(opt)}/{mode}/{generate_save_path_2(opt)}'
    return save_pth



def get_optimizer(optimizer_name, model_parameters, learning_rate=0.1, momentum=0.9, weight_decay=1e-4):
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = AdamW(model_parameters, lr=learning_rate)
    else:
        raise ValueError("'sgd' or 'adam'.")

    return optimizer

import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(scheduler_name, optimizer, decay_epochs=1, decay_factor=0.1, t_max=50):
    """
    Get the specified learning rate scheduler instance.

    Parameters:
        scheduler_name (str): The name of the scheduler, can be 'step' or 'cosine'.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        decay_epochs (int): Number of epochs for each decay period, used for StepLR scheduler (default is 1).
        decay_factor (float): The factor by which the learning rate will be reduced after each decay period,
                             used for StepLR scheduler (default is 0.1).
        t_max (int): The number of epochs for the cosine annealing scheduler (default is 50).

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The instance of the selected scheduler.
    
    """
    if isinstance(optimizer, torch.optim.Adam):
        return DummyScheduler()
    if scheduler_name.lower() == 'dummy':
        return DummyScheduler()
    if scheduler_name.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=decay_factor)
    elif scheduler_name.lower() == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name.lower() == "multi_step":
        decay_epochs = [150, 225]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step150":
        decay_epochs = [50, 100]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step2":
        decay_epochs = [40, 80]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step_imagenet":
        decay_epochs = [30, 60]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step_wide_resnet":
        decay_epochs = [60,120,160]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.2)
    else:
        raise ValueError("Unsupported scheduler name.")

    return scheduler

class DummyScheduler:
    def step(self):
        pass
def phi_stable_batch_epsilon( probs, labels, epsilon=1e-10):
    posterior_probs = probs + epsilon

    one_hot_labels = torch.zeros_like(posterior_probs)
    one_hot_labels[torch.arange(labels.size(0)),labels] = 1

    log_likelihood_correct = torch.log(posterior_probs[torch.arange(labels.size(0)), labels])
    sum_incorrect = torch.sum(posterior_probs * (1 - one_hot_labels), dim=1)
    sum_incorrect = torch.clamp(sum_incorrect, min=epsilon)

    log_likelihood_incorrect = torch.log(sum_incorrect)
    phi_stable = log_likelihood_correct - log_likelihood_incorrect

    return phi_stable
def cross_entropy(prob, label):
    epsilon = 1e-12
    prob = np.clip(prob, epsilon, 1.0 - epsilon)
    one_hot_label = np.zeros_like(prob)
    one_hot_label[np.arange(len(label)), label] = 1
    return -np.sum(one_hot_label * np.log(prob), axis=1)

def compute_cross_entropy_losses(data, model, device, batch_size=512):
    """
    Compute cross-entropy losses for each sample in a given dataset or dataloader.

    :param data: PyTorch Dataset or DataLoader containing the data.
    :param model: PyTorch model for making predictions.
    :param batch_size: Batch size to use for the DataLoader if data is a Dataset.
    :return: A list of losses, one for each sample.
    """
    model.to(device)
    # Ensure the model is in evaluation mode
    model.eval()
    # Check if data is already a DataLoader
    if isinstance(data, DataLoader):
        dataloader = data
    elif isinstance(data, Dataset):
        # Create DataLoader for loading the data
        dataloader = DataLoader(data, batch_size=batch_size)
    else:
        raise TypeError("data must be a PyTorch Dataset or DataLoader object")
    # List to store loss for each sample
    losses = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, targets in dataloader:
            # Compute the model output
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Compute cross-entropy loss
            loss = F.cross_entropy(outputs, targets, reduction='none')
            losses.extend(loss.tolist())

    return losses


def compute_phi_stable(data, model, device, batch_size=512):
    """
    Compute cross-entropy losses for each sample in a given dataset or dataloader.

    :param data: PyTorch Dataset or DataLoader containing the data.
    :param model: PyTorch model for making predictions.
    :param batch_size: Batch size to use for the DataLoader if data is a Dataset.
    :return: A list of losses, one for each sample.
    """
    model.to(device)
    # Ensure the model is in evaluation mode
    model.eval()
    # Check if data is already a DataLoader
    if isinstance(data, DataLoader):
        dataloader = data
    elif isinstance(data, Dataset):
        # Create DataLoader for loading the data
        dataloader = DataLoader(data, batch_size=batch_size)
    else:
        raise TypeError("data must be a PyTorch Dataset or DataLoader object")
    # List to store loss for each sample
    phi_stables = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, targets in dataloader:
            # Compute the model output
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs)
            phi_stable = phi_stable_batch_epsilon(probs,targets )
            # Compute cross-entropy loss

            phi_stables.extend(phi_stable.tolist()) 
    return phi_stables







def calculate_entropy(data_loader, model, device):
    entropies = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            
            probabilities = F.softmax(outputs, dim=1)
            
            entropy = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=1)    
            entropies.extend(entropy.cpu().numpy())
            #entropies.append(entropy)
            
    
    return entropies

def plot_entropy_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device):
    # Calculate entropies for target_train_loader and target_test_loader
    #target_train_loader = [(data.to(device), target.to(device)) for data, target in target_train_loader]
    #target_test_loader = [(data.to(device), target.to(device)) for data, target in target_test_loader]
    
    
    train_entropies = calculate_entropy(target_train_loader, target_model, device)
    test_entropies = calculate_entropy(target_test_loader, target_model, device)
    train_mean = np.mean(train_entropies)
    train_variance = np.var(train_entropies)
    test_mean = np.mean(test_entropies)
    test_variance = np.var(test_entropies)
    dict_sta = {"entropies_train_mean":train_mean, "entropies_train_variance": train_variance,
            "entropies_test_mean": test_mean, "entropies_test_variance" :test_variance}
    
    store_dict_to_yaml(dict_sta, save_path,"entropy_distribution.yaml")
    
    print(f'Entropies: train_mean:{train_mean: .3f} train_variance:{train_variance: .3f} test_mean:{test_mean: .3f} test_variance:{test_variance: .3f}')
    
    # Plot the distribution of entropies
    plt.figure(figsize=(8, 6))
    plt.hist(train_entropies, bins=50, alpha=0.5,range= (0,5), label=f'Train Entropy\nMean: {train_mean:.2f}\nVariance: {train_variance:.2f}', color='blue')
    plt.hist(test_entropies, bins=50, alpha=0.5,range= (0,5), label=f'Test Entropy\nMean: {test_mean:.2f}\nVariance: {test_variance:.2f}', color='red')
    plt.xlabel('Entropy') 
    plt.ylabel('Frequency')
    plt.title('Entropy Distribution for Target Data')
    plt.legend()
    plt.grid(True)
    save_path = f'{save_path}/entropy_distribution_comparison.png'
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    gc.collect()
    torch.cuda.empty_cache()
    
def plot_phi_distribution_together(data_in, data_out, save_path,name ="phi_distribution_target_comparison"):
    # Calculate entropies for target_train_loader and target_test_loader
    #target_train_loader = [(data.to(device), target.to(device)) for data, target in target_train_loader]
    #target_test_loader = [(data.to(device), target.to(device)) for data, target in target_test_loader]
    

    train_mean = np.mean(data_in)
    train_variance = np.var(data_in)
    test_mean = np.mean(data_out)
    test_variance = np.var(data_out)
    dict_sta = {"phi_train_mean":train_mean, "phi_train_variance": train_variance,
            "phi_test_mean": test_mean, "phi_test_variance" :test_variance}
    
    store_dict_to_yaml(dict_sta, save_path,"phi_distribution.yaml")
    
    print(f'Phi: train_mean:{train_mean: .3f} train_variance:{train_variance: .3f} test_mean:{test_mean: .3f} test_variance:{test_variance: .3f}')
    
    # Plot the distribution of entropies 
    plt.figure(figsize=(8, 6))
    plt.hist(data_in, bins=50, alpha=0.5,range= (-10,15), label=f'Train phi\nMean: {train_mean:.2f}\nVariance: {train_variance:.2f}', color='blue')
    plt.hist(data_out, bins=50, alpha=0.5,range= (-10,15), label=f'Test phi\nMean: {test_mean:.2f}\nVariance: {test_variance:.2f}', color='red')
    plt.xlabel(r'$\phi(p_y)$') 
    plt.ylabel('Frequency')
    plt.title(r'$\phi(p_y)$ distribution for target data')
    plt.legend()
    plt.grid(True)
    save_path = f'{save_path}/{name}.png'
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    gc.collect()
    
    
def plot_celoss_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device):
    # Calculate loss for target_train_loader and target_test_loader
    #target_train_loader = [(data.to(device), target.to(device)) for data, target in target_train_loader]
    #target_test_loader = [(data.to(device), target.to(device)) for data, target in target_test_loader]
    
    
    train_loss = compute_cross_entropy_losses(target_train_loader, target_model, device)
    test_loss = compute_cross_entropy_losses(target_test_loader, target_model, device)
    train_mean = np.mean(train_loss)
    train_variance = np.var(train_loss)
    test_mean = np.mean(test_loss)
    test_variance = np.var(test_loss)
    
    dict_sta = {"loss_train_mean":train_mean, "loss_train_variance": train_variance,
            "loss_test_mean": test_mean, "loss_test_variance" :test_variance}
    
    store_dict_to_yaml(dict_sta, save_path,"loss_distribution.yaml")
    print(f'Loss: train_mean:{train_mean: .8f} train_variance:{train_variance: .8f} test_mean:{test_mean: .8f} test_variance:{test_variance: .8f}')
    # Plot the distribution of entropies

    plt.figure(figsize=(8, 6))
    plt.hist(train_loss, bins=50, range= (0,5),alpha=0.5, label=f'Train Loss\nMean: {train_mean:.2f}\nVariance: {train_variance:.2f}', color='blue')
    plt.hist(test_loss, bins=50,range= (0,5), alpha=0.5, label=f'Test Loss\nMean: {test_mean:.2f}\nVariance: {test_variance:.2f}', color='red')
    plt.xlabel('Loss') 
    plt.ylabel('Frequency')
    plt.title('Loss Distribution for Target Data')
    plt.legend()
    plt.grid(True)
    save_path = f'{save_path}/loss_distribution_comparison.png'
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    gc.collect()
    torch.cuda.empty_cache()
    


import torchvision
import torch.nn as nn
import torch.hub

def get_target_model(name="resnet18", num_classes=10, dropout=None, fintune = False):
    if name == "resnet18":
        model = torchvision.models.resnet18(weights= fintune)
        num_ftrs = model.fc.in_features
        if dropout is not None:
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == "resnet20":
        model = resnet20(num_classes=num_classes)
        if dropout is not None:
            pass
    elif name == "resnet34":
        if fintune:
            model = torchvision.models.resnet34(weights= "default")
        else:
            model = torchvision.models.resnet34(weights= None)
        num_ftrs = model.fc.in_features
        if dropout is not None:
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights= fintune)
        num_ftrs = model.fc.in_features
        if dropout is not None:
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == "vgg11":
        model = torchvision.models.vgg11(weights= fintune)
        num_ftrs = model.classifier[-1].in_features
        if dropout is not None:
            model.classifier[-1] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif name == "wide_resnet50":
        model = torchvision.models.wide_resnet50_2(weights= fintune)
        num_ftrs = model.fc.in_features
        if dropout is not None:
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == "densenet121":
        if fintune:
            model = torchvision.models.densenet121(weights= "default")
        else:
            model = torchvision.models.densenet121(weights= None)
        num_ftrs = model.classifier.in_features
        if dropout is not None:
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)
            
    elif name == "vit":
        if fintune:
            model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        else: model = torchvision.models.vit_b_16()
        num_ftrs = model.heads.head.in_features
        if dropout is not None:
            model.heads.head = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Dropout(dropout)
            )
        else:
            model.heads.head = nn.Linear(num_ftrs, num_classes)
            
    elif name == "TexasClassifier":
        model= Texas(num_classes = num_classes, droprate = dropout)
    elif name == "PurchaseClassifier":
        model= Purchase(num_classes = num_classes, droprate = dropout)

    else:
        raise ValueError("Model not implemented yet :P")

    return model


def get_dropout_fc_layers(model, rate = 0.5):
    last_layer =list(model.children())[-1]
    if isinstance(last_layer, nn.Sequential):
        last_layer = last_layer[-1]
    dropout_layer = nn.Dropout(p=rate)
    new_last_layer = nn.Sequential(dropout_layer,last_layer) 
    return new_last_layer


def add_new_last_layer(model, new_last_layer):
    if hasattr(model, 'fc'):
        if isinstance(model.fc, nn.Sequential):
            model.fc[-1] =new_last_layer
        
        else: model.fc = new_last_layer
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] =new_last_layer
        else: model.classifier = new_last_layer
    else:
        raise AttributeError("there is no 'fc' or 'classifier'")

def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)

def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        # multi-class, one-hot encoded
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
                labels = np.expand_dims(labels, axis=1)
        elif (
            len(
                labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2
        ):  # multi-class, index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
            else:
                labels = np.expand_dims(labels, axis=1)
        elif (
            len(
                labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2
        ):  # binary, index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 1:  # index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
            else:
                labels = np.expand_dims(labels, axis=1)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels



