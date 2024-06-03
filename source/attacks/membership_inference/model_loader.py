import os
import torch
from utils import get_target_model, generate_save_path

class ModelLoader:
    def __init__(self, args, mode="target"):
        self.args = args
        self.mode = mode
        self.load_path = self.generate_load_path()  # Call the method to set load_path on initialization
        

    def generate_load_path(self):
        if self.args.specific_path:
            return self.args.load_model_path
        else:
            return generate_save_path(self.args, mode=self.mode)

    def get_load_model_path(self):
        if self.args.training_type == "DPSGD":
            return os.path.join(self.load_path, f"{self.args.model}.pt")
        else:
            return os.path.join(self.load_path, f"{self.args.model}.pth")

    def get_model_framework(self):
        if self.args.training_type == "Dropout":
            return get_target_model(name=self.args.model, num_classes=self.args.num_class, dropout=self.args.tau)
        else:
            return get_target_model(name=self.args.model, num_classes=self.args.num_class)

    def get_loaded_model(self, load_model_path):
        if self.args.training_type == "DPSGD":
            return torch.load(load_model_path)
        else:
            model = self.get_model_framework()
            model.load_state_dict(torch.load(load_model_path, map_location=self.args.device))
            return model

    def __call__(self):
        load_model_path = self.get_load_model_path()
        model = self.get_loaded_model(load_model_path)
        model.to(self.args.device)
        model.eval()
        return model

class ShadowModelLoader(ModelLoader):
    def __init__(self, args, mode="shadow"):
        super().__init__(args, mode)
        self.shadow_split_num = args.shadow_split_num
    def __call__(self):
        models = []
        # You need to ensure that the original mode in self is not changed permanently
        original_mode = self.mode
        for index in range(self.shadow_split_num):
            self.mode = f"shadow_{index}"  # Temporarily set the mode for this iteration
            self.load_path = self.generate_load_path()  # Generate the path for the current mode
            model = super().__call__()  # Load the model
            models.append(model)
            self.mode = original_mode  # Reset the mode to the original after use
        return models
