import torchvision
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from source.utility.main_parse import add_argument_parameter
from attacks.membership_inference.data_augmentation_attack import AugemtaionAttackDataset, DataAugmentationMIA
from source.attacks.membership_inference.attack_dataset import AttackDataset
from source.attacks.membership_inference.black_box_attack import BlackBoxMIA
from source.attacks.membership_inference.label_only_attack import LabelOnlyMIA
from source.attacks.membership_inference.metric_based_attack import MetricBasedMIA
import torch
from data_preprocessing.data_loader_target import BuildDataLoader
from utils import get_target_model, generate_save_path, plot_celoss_distribution_together
import argparse
import numpy as np
import os
torch.set_num_threads(1)
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)



def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--load-pretrained', type=str, default='no')
    #--training type there is used for specifying path to load model
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "metric-based", and "label-only"')
    add_argument_parameter(parser)
    args = parser.parse_args()
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    return args


def check_loss_distr(save_path,file_name= "loss_distribution.yaml"):
    yaml_2 = YAML()
    loss_distr = f"{save_path}/{file_name}"
    if os.path.exists(loss_distr):
        with open(loss_distr, 'r') as f:
            #distribution = yaml.safe_load(f)
            distribution = yaml_2.load(f)
            return not isinstance(distribution["loss_train_mean"], ScalarFloat)
    else:
        return True
    
    
import torch

def get_image_shape(dataloader):
    try:
        # Get a batch of data
        data, labels = next(iter(dataloader))
        
        # Check the shape of a single image
        image_shape = data[0].shape
        
        return image_shape
    except StopIteration:
        print("Data loader is empty or exhausted.")
        return None

if __name__ == "__main__":
    args = parse_args()
    device = args.device
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)
    s = BuildDataLoader(args)
    
    if args.inference:
        if args.attack_type == "augmentation":
            target_train, target_test, _, shadow_train, shadow_test = s.get_data_supervised_inference(batch_size =args.batch_size, num_workers =args.num_workers, if_dataset=True)
        else:
            target_train_loader, target_test_loader, _,shadow_train_loader, shadow_test_loader  = s.get_data_supervised_inference(batch_size =args.batch_size, num_workers =args.num_workers)
        
    else:
        if args.attack_type == "augmentation":
            target_train, target_test, shadow_train, shadow_test = s.get_data_supervised_ni(batch_size =args.batch_size, num_workers =args.num_workers, if_dataset=True)
        else:  
            target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader  = s.get_data_supervised_ni(batch_size =args.batch_size, num_workers =args.num_workers)


    target_model = get_target_model(name=args.model, num_classes=args.num_class)
    shadow_model = get_target_model(name= args.model, num_classes=args.num_class)

    temp_save = str(args.temp).rstrip('0').rstrip('.') if '.' in str(args.temp) else str(args.temp)

    
    if args.specific_path:
        load_path_target = f"{args.load_model_path}/{args.model}.pth"
        load_path_shadow = load_path_target.replace("/target/", "/shadow/")
        save_path = args.load_model_path
    else:
        load_path_target = f'{generate_save_path(args, mode = "target")}/{args.model}.pth'
        load_path_shadow = f'{generate_save_path(args, mode = "shadow")}/{args.model}.pth'
        save_path = generate_save_path(args, mode = "target")
    # load target/shadow model to conduct the attacks

    target_model.load_state_dict(torch.load(load_path_target, map_location=args.device))
    shadow_model.load_state_dict(torch.load(load_path_shadow, map_location=args.device))
    


    target_model = target_model.to(args.device)
    target_model.eval()
    shadow_model = shadow_model.to(args.device)
    shadow_model.eval()
    attack_type = args.attack_type

    if attack_type != "augmentation":
        input_shape = get_image_shape(target_train_loader)
    if attack_type == "augmentation":
        attack_dataset_rotation = AugemtaionAttackDataset( args, "rotation" , target_model, shadow_model,
                                        target_train, target_test, shadow_train, shadow_test,device)
        
        attack_dataset_translation =AugemtaionAttackDataset( args, "translation" , target_model, shadow_model,
                                        target_train, target_test, shadow_train, shadow_test,device)
        print(attack_dataset_rotation.attack_train_dataset.data.shape[1])
        print("Attack datasets are ready")
    else:
        attack_dataset = AttackDataset(args, attack_type, target_model, shadow_model,
                                        target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader)


        # train attack model

    if "black_box" in attack_type or "black-box" in attack_type:
        attack_model = BlackBoxMIA(
            num_class=args.num_class,
            device=args.device,
            attack_type=attack_type,
            attack_train_dataset=attack_dataset.attack_train_dataset,
            attack_test_dataset=attack_dataset.attack_test_dataset,
            save_path = save_path,
            batch_size=128)
    elif "metric-based" in attack_type:

        attack_model = MetricBasedMIA(
            args = args,
            num_class=args.num_class,
            device=args.device,
            attack_type=attack_type,
            attack_train_dataset=attack_dataset.attack_train_dataset,
            attack_test_dataset=attack_dataset.attack_test_dataset,
            #train_loader = target_train_loader,
            save_path = save_path,
            batch_size=128)
    elif "augmentation" in attack_type:
        attack_model = DataAugmentationMIA(
            num_class = attack_dataset_rotation.attack_train_dataset.data.shape[1],
            device = args.device, 
            attack_type= "rotation",
            attack_train_dataset=attack_dataset_rotation.attack_train_dataset,  
            attack_test_dataset= attack_dataset_rotation.attack_train_dataset,  
            save_path= save_path, 
            batch_size= 128)
        attack_model = DataAugmentationMIA(
            num_class = attack_dataset_translation.attack_train_dataset.data.shape[1],
            device = args.device, 
            attack_type= "translation",
            attack_train_dataset=attack_dataset_translation.attack_train_dataset,  
            attack_test_dataset= attack_dataset_translation.attack_test_dataset,
            save_path= save_path, 
            batch_size= 128)
    else: raise ValueError("No attack is executed")