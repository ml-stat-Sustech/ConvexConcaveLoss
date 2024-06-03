import copy
import os
import numpy as np
import torch

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_teacher_model(model,teacher_path ,device):
    """
    Load a teacher model from the given path.
    Returns:
        MyModel: The loaded model.
    """
    # Ensure the model path exists
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"No model found at {teacher_path}")
    # Move model to appropriate device
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    # Load the state dict into the model
    model_copy.load_state_dict(torch.load(teacher_path, map_location=device))
    # Set the model to evaluation mode
    model_copy.eval()
    return model_copy

def freeze_except_last_layer(model):
    """
    Freeze the parameters of all layers in the model except the last one.
    Args:
        model (nn.Module): The model to freeze the parameters of.
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last layer
    for param in list(model.children())[-1].parameters():
        param.requires_grad = True
