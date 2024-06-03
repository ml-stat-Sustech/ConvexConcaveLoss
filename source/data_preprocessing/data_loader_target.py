

import os
import torchvision.transforms as transforms
import torch
import numpy as np
from io import RawIOBase

from data_preprocessing.data_no_image import prepare_purchase, prepare_texas
from data_preprocessing.dataset_preprocessing import prepare_dataset, cut_dataset, prepare_dataset_ni, \
    prepare_inference_dataset, prepare_dataset_target, prepare_dataset_inference, prepare_dataset_shadow_splits
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from data_preprocessing import configs
import torchvision


class BuildDataLoader(object):
    def __init__(self, args,shuffle= True):
        self.args = args
        self.data_path = args.data_path
        self.input_shape = args.input_shape
        self.batch_size = args.batch_size
        self.num_splits = args.shadow_split_num
        self.shuffle = shuffle
    def parse_dataset(self, dataset, train_transform, test_transform):

        if dataset.lower() == "imagenet":
            self.data_path = f'{self.data_path}/images/'
            train_dataset = torchvision.datasets.ImageFolder(root=self.data_path + 'train', transform=train_transform)
            test_dataset = torchvision.datasets.ImageFolder(root=self.data_path + 'val', transform=test_transform)
            dataset = train_dataset + test_dataset
            
            return dataset
        
        if dataset.lower() == "tinyimagenet":
            #self.data_path = "/data1/lzl/tiny-imagenet-200/"
            self.data_path = f'{self.data_path}/tiny-imagenet-200/'
            image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(self.data_path, x), transform=train_transform)
                  for x in ['train', 'val','test']}
            dataset =  image_datasets['train'] + image_datasets['val'] +image_datasets['test']
            return dataset
        
        if dataset.lower() == "imagenet_r":
            self.data_path = "/data/dataset/imagenet-rendition/imagenet-r/"
            dataset = torchvision.datasets.ImageFolder(root=self.data_path, transform=train_transform)
            return dataset
        if dataset.lower() == "purchase":
            dataset = prepare_purchase(self.data_path)
            return dataset
        if dataset.lower() == "texas":
            dataset = prepare_texas(self.data_path)
            return dataset
        if dataset in configs.SUPPORTED_IMAGE_DATASETS:
            _loader = getattr(datasets, dataset)
            if dataset != "EMNIST":
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       transform=test_transform,
                                       download=True)
            else:
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        split="byclass",
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       split="byclass",
                                       transform=test_transform,
                                       download=True)
            dataset = train_dataset + test_dataset

        else:
            raise ValueError("Dataset Not Supported: ", dataset)
        return dataset

    def get_data_transform(self, dataset, use_transform="simple"):
        
        if dataset.lower() in ["imagenet", "imagenet_r"]:
            transform_list = [transforms.Resize(256),transforms.CenterCrop(224)]
            #transform_list = [transforms.RandomResizedCrop(224)]
            if use_transform == "simple":
                transform_list += [transforms.RandomHorizontalFlip()]
                print("add simple data augmentation!")
            transform_list+= [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            transform_ = transforms.Compose(transform_list)
            return transform_
        
        if dataset.lower() in ["tinyimagenet"]:
            if self.args.finetune:
                transform_list = [transforms.RandomResizedCrop(224)]
            else: transform_list = [transforms.RandomResizedCrop(64)]
            if use_transform == "simple":
                transform_list += [transforms.RandomHorizontalFlip()]
                print("add simple data augmentation for tinyimagenet!")
            transform_list+= [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            transform_ = transforms.Compose(transform_list)
            return transform_
        transform_list = [transforms.Resize(
            (self.input_shape[0], self.input_shape[0])), ]
        if use_transform == "simple":
            transform_list += [transforms.RandomCrop(
                32, padding=4), transforms.RandomHorizontalFlip(), ]
            print("add simple data augmentation!")
        transform_list.append(transforms.ToTensor())
        if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
            transform_list = [
                transforms.Grayscale(3), ] + transform_list
        transform_ = transforms.Compose(transform_list)
        return transform_


    def get_dataset(self, train_transform, test_transform):
        """
        The function "get_dataset" returns a parsed dataset using the specified train and test
        transformations.
        
        :param train_transform: train_transform is a transformation function that is applied to the
        training dataset. It can include operations such as data augmentation, normalization, resizing,
        etc. This function is used to preprocess the training data before it is fed into the model for
        training
        :param test_transform: The `test_transform` parameter is a transformation that is applied to the
        test dataset. It is used to preprocess or augment the test data before it is fed into the model
        for evaluation. This can include operations such as resizing, normalization, or data
        augmentation techniques like random cropping or flipping
        :return: The dataset is being returned.
        """
        dataset = self.parse_dataset(
            self.args.dataset, train_transform, test_transform)
        return dataset



    def get_data_supervised(self, num_workers=2, select_num=None):
        batch_size = self.batch_size
        
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_test = prepare_dataset_target(
            dataset, select_num=select_num)

        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        print("target_train: %d  \t target_test: %s" %
            (len(target_train), len(target_test)))

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        

        return target_train_loader, target_test_loader
    def get_data_supervised_inference(self, batch_size=128, num_workers=8, select_num=None, if_dataset =False):
        # inference 1/5
        # self.args.dataset default CIFAR10
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_test,inference, shadow_train, shadow_test = prepare_dataset_inference(
            dataset, select_num=select_num)

        if if_dataset:
            print("Return inference dataset")
            return target_train, target_test, inference, shadow_train, shadow_test
        
        
        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        print("target_train: %d \t target_test: %s inference_dataset: %s"  %
            (len(target_train), len(target_test), len(inference)))

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        inference_data_loader = torch.utils.data.DataLoader(
            inference, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        return target_train_loader, target_test_loader,inference_data_loader, shadow_train_loader, shadow_test_loader
    def get_split_shadow_dataset_ni(self, select_num=None, if_dataset =False, num_splits =16,shadow_datapoint_num =None):
        # inference 1/5
        # self.args.dataset default CIFAR10
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        _, _,  shadow_train, shadow_test = prepare_dataset_ni(
            dataset, select_num=select_num)
        if shadow_datapoint_num is not None:
            split_size = shadow_datapoint_num
        else:
            split_size = len(shadow_train)
        shadow_list = prepare_dataset_shadow_splits(dataset = shadow_train+ shadow_test, num_splits= num_splits, split_size= split_size)# list[tuple]
        print(f"Prepare shadow dataset list, total num of the list: {num_splits}")
        self.shadow_dataset_list = shadow_list
        if if_dataset:
            return shadow_list

    def get_split_shadow_dataloader_ni(self, batch_size=128, num_workers=8,index= 0):
        train_dataset, test_dataset= self.shadow_dataset_list[index]
        print("Preparing dataloader!")
        print(f"shadow dataset index: {index}")
        print(f"train: {len(train_dataset)} \t target_test: {len(test_dataset)}")

        shadow_train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        shadow_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        return shadow_train_loader, shadow_test_loader

    def get_split_shadow_dataset_inference(self, select_num=None, if_dataset =False, num_splits =16,shadow_datapoint_num =None):
        # inference 1/5
        # self.args.dataset default CIFAR10
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        _, _, inference, shadow_train, shadow_test = prepare_dataset_inference(
            dataset, select_num=select_num)
        if shadow_datapoint_num is not None:
            split_size = shadow_datapoint_num
        else:
            split_size = len(shadow_train)+len(shadow_test)
        shadow_list = prepare_dataset_shadow_splits(dataset = shadow_train+ shadow_test, num_splits= num_splits, split_size= split_size)
        print(f"Prepare shadow dataset list, total num of the list: {num_splits}")
        self.inference_dataset = inference
        self.shadow_dataset_list = shadow_list
        if if_dataset:
            return inference, shadow_list
    def get_split_shadow_dataloader_inference(self, batch_size=128, num_workers=8,index= 0):

        #self.get_split_shadow_dataset_inference(select_num=None, num_splits= num_splits)

        train_dataset, test_dataset= prepare_dataset_target(self.shadow_dataset_list[index])

        print("Preparing dataloader!")
        print(f"shadow dataset index: {index}")
        print("train: %d \t target_test: %s inference_dataset: %s" %
              (len(train_dataset), len(test_dataset), len(self.inference_dataset)))


        inference_data_loader = torch.utils.data.DataLoader(
            self.inference_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        shadow_train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        shadow_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        return inference_data_loader, shadow_train_loader, shadow_test_loader
    
    
    
    
    def get_data_supervised_ni(self, batch_size=128, num_workers=2, select_num=None, if_dataset = False):
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_test, shadow_train, shadow_test = prepare_dataset_ni(
            dataset, select_num=select_num)
        if if_dataset:
            print("Return dataset")
            return target_train, target_test, shadow_train, shadow_test
        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        print("target_train: %d \t target_test: %s" %
            (len(target_train), len(target_test)))

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        return target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader

    
    def get_ordered_dataset(self, target_dataset):
        """
    
        Sorts and returns a dataset based on the labels of the data points.

        Parameters:
        - target_dataset (Dataset): The dataset to be sorted.

        Returns:
        - Subset: The sorted dataset.

        Inspired by https://stackoverflow.com/questions/66695251/define-manually-sorted-mnist-dataset-with-batch-size-1-in-pytorch
        """
        label = np.array([row[1] for row in target_dataset])
        sorted_index = np.argsort(label)
        sorted_dataset = torch.utils.data.Subset(target_dataset, sorted_index)
        return sorted_dataset

    def get_label_index(self, target_dataset):
        """
        return starting index for different labels in the sorted dataset
        """
        label_index = []
        start_label = 0
        label = np.array([row[1] for row in target_dataset])
        for i in range(len(label)):
            if label[i] == start_label:
                label_index.append(i)
                start_label += 1
        return label_index
    
    def get_sorted_data_mixup_mmd_one_inference(self):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, _,  inference, shadow_train, _ = prepare_dataset_inference(
            dataset, select_num=None)
            # sort by label
        target_train_sorted = self.get_ordered_dataset(target_train)
        target_inference_sorted = self.get_ordered_dataset(inference) # dataset
        shadow_train_sorted = self.get_ordered_dataset(shadow_train)
        shadow_inference_sorted = self.get_ordered_dataset(inference)
 

        start_index_target_inference = self.get_label_index(
            target_inference_sorted)
        start_index_shadow_inference = self.get_label_index(
            shadow_inference_sorted)

        # note that we set the inference loader's batch size to 1
        target_train_sorted_loader = torch.utils.data.DataLoader(
            target_train_sorted, batch_size=self.args.batch_size, shuffle=self.shuffle, num_workers=self.args.num_workers, pin_memory=True)
        target_inference_sorted_loader = torch.utils.data.DataLoader(
            target_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_train_sorted_loader = torch.utils.data.DataLoader(
            shadow_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_inference_sorted_loader = torch.utils.data.DataLoader(
            shadow_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        return target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted
    
    
    def get_sorted_data_mixup_mmd(self):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test = prepare_dataset(
            dataset, select_num=None)
            # sort by label
        target_train_sorted = self.get_ordered_dataset(target_train)
        target_inference_sorted = self.get_ordered_dataset(target_inference) # dataset
        shadow_train_sorted = self.get_ordered_dataset(shadow_train)
        shadow_inference_sorted = self.get_ordered_dataset(shadow_inference)
 

        start_index_target_inference = self.get_label_index(
            target_inference_sorted)
        start_index_shadow_inference = self.get_label_index(
            shadow_inference_sorted)

        # note that we set the inference loader's batch size to 1
        target_train_sorted_loader = torch.utils.data.DataLoader(
            target_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        target_inference_sorted_loader = torch.utils.data.DataLoader(
            target_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_train_sorted_loader = torch.utils.data.DataLoader(
            shadow_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_inference_sorted_loader = torch.utils.data.DataLoader(
            shadow_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        return target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted
# target_train_sorted_loader  target_inference_sorted_loader start_index_target_inference target_inference_sorted

class GetDataLoaderPoison(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.input_shape = args.input_shape
    def parse_dataset(self, dataset):
        if dataset in configs.SUPPORTED_IMAGE_DATASETS:
            _loader = getattr(datasets, dataset)
            if dataset != "EMNIST":
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        transform=None,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       transform=None,
                                       download=True)
            else:
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        split="byclass",
                                        transform=None,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       split="byclass",
                                       transform=None,
                                       download=True)

        else:
            raise ValueError("Dataset Not Supported: ", dataset)
        return train_dataset, test_dataset
    def get_data_transform(self, dataset, use_transform="simple"):
        transform_list = [transforms.Resize(
            (self.input_shape[0], self.input_shape[0])), ]
        if use_transform == "simple":
            transform_list += [transforms.RandomCrop(
                32, padding=4), transforms.RandomHorizontalFlip(), ]
            print("add simple data augmentation!")

        transform_list.append(transforms.ToTensor())

        if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
            transform_list = [
                transforms.Grayscale(3), ] + transform_list

        transform_ = transforms.Compose(transform_list)
        return transform_

    def get_data_loader(self):

        train_transform = self.get_data_transform(
            self.args.dataset, use_transform=None)
        test_transform = self.get_data_transform(
            self.args.dataset, use_transform=None)

        train_dataset, test_dataset = self.parse_dataset(self.args.dataset)

        bad_params = {
            'trigger': self.args.trigger,
            'attack': self.args.attack,
            'src_label': self.args.atk_src_label,
            'tar_label': self.args.atk_tar_label,
        }

        # get train data
        bad_train_indices, trigger_trans, attack_trans = prepare_backdoor_attack(in_size=self.input_shape[0],
                                                                                 classes=self.args.num_class,
                                                                                 labels=train_dataset.targets,
                                                                                 proportion=self.args.atk_proportion,
                                                                                 **bad_params)
        clr_trainset = BackdoorDataset(train_dataset, train_transform)
        bad_trainset = BackdoorDataset(
            train_dataset, train_transform, bad_train_indices, trigger_trans, attack_trans)

        # get test data
        bad_test_indices, trigger_trans, attack_trans = prepare_backdoor_attack(in_size=self.input_shape[0],
                                                                                classes=self.args.num_class,
                                                                                labels=test_dataset.targets,
                                                                                proportion=1.0,
                                                                                **bad_params)

        clr_testset = BackdoorDataset(test_dataset, test_transform)
        bad_testset = BackdoorDataset(
            test_dataset, test_transform, bad_test_indices, trigger_trans, attack_trans)

        bad_testset = IndicesDataset(bad_testset, bad_test_indices)
        # print(bad_test_indices)
        # print(bad_testset[0])
        # print(len(bad_testset))
        # exit()

        # generate dataloader
        bad_trainloader = torch.utils.data.DataLoader(
            bad_trainset, batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=self.args.num_workers)

        clr_testloader = torch.utils.data.DataLoader(
            clr_testset,
            batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=self.args.num_workers)

        bad_testloader = torch.utils.data.DataLoader(
            bad_testset,
            batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=self.args.num_workers)

        return bad_trainloader, clr_testloader, bad_testloader


def prepare_backdoor_attack(
        in_size: int,
        classes: int,
        labels: list,
        proportion: float = 0.0,
        trigger: str = None,
        attack: str = None,
        src_label: int = None,
        tar_label: int = None) -> tuple:

    if (trigger is None) or (attack is None):
        return None, None, None

    mask = np.ones([in_size, in_size], dtype=np.uint8)
    pattern = np.zeros([in_size, in_size], dtype=np.uint8)

    if trigger == 'single-pixel':
        mask[-2, -2], pattern[-2, -2] = 0, 255

    elif trigger == 'pattern':
        mask[-2, -2], pattern[-2, -2] = 0, 255
        mask[-2, -4], pattern[-2, -4] = 0, 255
        mask[-4, -2], pattern[-4, -2] = 0, 255
        mask[-3, -3], pattern[-3, -3] = 0, 255

    else:
        raise ValueError(
            'backdoor trigger {} is not supported'.format(trigger))

    trigger_trans = TriggerTrans(mask, pattern)

    if attack == 'single-target':
        indices = np.where(np.array(labels) == src_label)[0]
        attack_trans = SingleTargetTrans(src_label, tar_label)

    elif attack == 'all-to-all':
        indices = np.arange(len(labels))
        attack_trans = AlltoAllTrans(classes)

    elif attack == "all-to-single":
        indices = np.arange(len(labels))
        attack_trans = AlltoSingleTrans(tar_label)

    else:
        raise ValueError('backdoor attack {} is not supported'.format(attack))

    backdoor_num = int(len(indices) * proportion)
    backdoor_indices = np.random.permutation(indices)[:backdoor_num]

    return backdoor_indices, trigger_trans, attack_trans


class BackdoorDataset:
    def __init__(
            self, dataset, transform=None,
            backdoor_indices=None, trigger_trans=None, attack_trans=None):

        self.dataset = dataset
        self.transform = transform

        self.backdoor_indices = set() if (
            backdoor_indices is None) else set(backdoor_indices)
        self.trigger_trans = trigger_trans
        self.attack_trans = attack_trans

    def __getitem__(self, i):
        x, y = self.dataset[i]

        if i in self.backdoor_indices:
            x, y = self.trigger_trans(x), self.attack_trans(y)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)


class IndicesDataset:
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = range(len(dataset)) if (indices is None) else indices

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)


class TriggerTrans:
    def __init__(self, mask, pattern):
        self.mask = mask
        self.pattern = pattern

    def __call__(self, x):
        x = np.asarray(x)
        mask, pattern = self.mask, self.pattern

        if len(x.shape) == 3:
            mask = mask.reshape(*mask.shape, 1)
            pattern = pattern.reshape(*pattern.shape, 1)
        x_trigger = x * mask + pattern
        # change numpy array back to PIL.Image
        x_trigger = Image.fromarray(x_trigger)
        return x_trigger


class SingleTargetTrans:
    def __init__(self, src_label, tar_label):
        self.src_label = src_label
        self.tar_label = tar_label

    def __call__(self, y):
        assert y == self.src_label
        return self.tar_label


class AlltoAllTrans:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, y):
        return (y+1) % self.classes


class AlltoSingleTrans:
    def __init__(self, tar_label):
        self.tar_label = tar_label

    def __call__(self, y):
        return self.tar_label

