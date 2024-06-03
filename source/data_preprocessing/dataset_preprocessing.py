

import torch
import os

from torch.utils.data import Subset

torch.manual_seed(0)


def count_dataset(targetTrainloader, targetTestloader, shadowTrainloader, shadowTestloader, num_classes, attr=None):
    target_train = [0 for i in range(num_classes)]
    target_test = [0 for i in range(num_classes)]
    shadow_train = [0 for i in range(num_classes)]
    shadow_test = [0 for i in range(num_classes)]

    for _, num in targetTrainloader:
        if attr != None:
            num = num[attr]
        for row in num:
            target_train[int(row)] += 1

    for _, num in targetTestloader:
        if attr != None:
            num = num[attr]
        for row in num:
            target_test[int(row)] += 1

    for _, num in shadowTrainloader:
        if attr != None:
            num = num[attr]
        for row in num:
            shadow_train[int(row)] += 1

    for _, num in shadowTestloader:
        if attr != None:
            num = num[attr]
        for row in num:
            shadow_test[int(row)] += 1

    print(target_train)
    print(target_test)
    print(shadow_train)
    print(shadow_test)

"""
def prepare_dataset(dataset, select_num=None):
    
    length = len(dataset)
    each_length = length//6
    # if we specify a number, we use the number to split data
    if select_num != None and select_num < each_length:
        each_length = select_num
    # print(dataset.category_label_index_dict)
    torch.manual_seed(0)
    target_train, target_inference, target_test, shadow_train,  shadow_inference, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, each_length, each_length, len(dataset)-(each_length*6)])
    
    if select_num != None:
        target_train, target_inference, target_test, shadow_train,  shadow_inference, shadow_test, _ = torch.utils.data.random_split(
        dataset, select_num)
    return target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test
"""
def prepare_dataset(dataset, select_num=None):
    
    length = len(dataset)
    each_length = length//6
    # if we specify a number, we use the number to split data
    torch.manual_seed(0)
    if select_num is None:
        target_train, target_inference, target_test, shadow_train,  shadow_inference, shadow_test, _ = torch.utils.data.random_split(
            dataset, [each_length, each_length, each_length, each_length, each_length, each_length, len(dataset)-(each_length*6)])
    else:
        target_train, target_inference, target_test, shadow_train,  shadow_inference, shadow_test = torch.utils.data.random_split(
        dataset, select_num)
    # print(dataset.category_label_index_dict)
        
    return target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test

def prepare_dataset_ni(dataset, select_num=None):
    # no inference
    length = len(dataset)
    each_length = length//4
    # if we specify a number, we use the number to split data
    torch.manual_seed(0)
    if select_num is None:
        target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
            dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    else:
        target_train, target_test, shadow_train, shadow_test = torch.utils.data.random_split(
        dataset, select_num)
    # print(dataset.category_label_index_dict)
        
    return target_train, target_test, shadow_train, shadow_test

def prepare_dataset_inference(dataset, select_num=None):
    # no inference
    length = len(dataset)
    each_length = length//5
    # if we specify a number, we use the number to split data
    if select_num is None:
        target_train, target_test,inference, shadow_train, shadow_test, _ = torch.utils.data.random_split(
            dataset, [each_length, each_length, each_length, each_length, each_length, len(dataset)-(each_length*5)])
    else:
        target_train, target_test, inference, shadow_train, shadow_test = torch.utils.data.random_split(
        dataset, select_num)
    # print(dataset.category_label_index_dict)
        
    return target_train, target_test, inference, shadow_train, shadow_test


def prepare_dataset_target(dataset, select_num=None):

    if select_num is None:
        select_num = [0.5, 0.5]
    # print(dataset.category_label_index_dict)
    target_train, target_test= torch.utils.data.random_split(
        dataset, select_num)
    return target_train, target_test

def prepare_dataset_shadow_splits(dataset, num_splits, split_size = None, train_size_ratio =0.5):
    """
    Randomly splits the dataset into train and test parts multiple times, based on the given ratio.

    Args:
        dataset: The dataset to be split.
        num_splits: The number of times the dataset should be split.
        train_size_ratio: The proportion of the dataset to include in the train split.

    Returns:
        A list of tuples, each containing a train and test dataset split.
    """
    if split_size is None:
        total_length = len(dataset)
    else: total_length = split_size*2
    splits = []

    for _ in range(num_splits):
        train_size = int(total_length * train_size_ratio)
        test_size = total_length - train_size

        # Randomly split the dataset into train and test parts
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        splits.append((train_dataset, test_dataset))

    return splits




def get_target_shadow_dataset(dataset, target_size=None, shadow_size=None):
    if target_size:
        target_dataset, shadow_dataset = cut_dataset(dataset, target_size)
    elif shadow_size:
        shadow_dataset, target_dataset = cut_dataset(dataset, shadow_size)
    else:
        target_dataset, shadow_dataset = cut_dataset(dataset, len(dataset)//2)

    return target_dataset, shadow_dataset


def split_dataset(dataset, parts=3, part_size=None):
    length = len(dataset)
    each_length = length//parts
    # if we specify a number, we use the number to split data
    if part_size != None and part_size < each_length:
        each_length = part_size
    torch.manual_seed(0)
    train_, inference_, test_, _ = torch.utils.data.random_split(dataset,
                                                                 [each_length, each_length, each_length, len(dataset)-(each_length*parts)])
    return train_, inference_, test_


def prepare_inference_dataset(dataset):
    each_length = len(dataset) // 2
    torch.manual_seed(0)
    inference_train, inference_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, len(dataset)-(each_length*2)]
    )
    return inference_train, inference_test


def cut_dataset(dataset, num):

    length = len(dataset)

    torch.manual_seed(0)
    selected_dataset, _ = torch.utils.data.random_split(
        dataset, [num, length - num])
    return selected_dataset


def count_dataset_for_class(num_class, dataset):
    for label in range(num_class):
        indices = [i for i in range(len(dataset)) if dataset[i][1] == label]
        print("class: %d,  data num: %d" % (label, len(indices)))
