import os
import sys
import torch
import numpy as np
import urllib.request
import tarfile
#FILE_DIR = os.path.dirname(os.path.abspath(__file__))
#DATA_ROOT = os.path.join(FILE_DIR, '../../data')
#sys.path.append(os.path.join(FILE_DIR, '../'))

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def prepare_texas(data_root):
    '''
    Texas dataset:
    X: (67330,6169) binary feature
    Y: (67330,)  num_classes=100
    '''
    ## Dataset directory
    DATASET_PATH = os.path.join(data_root, 'texas')
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    DATASET_FEATURES = os.path.join(DATASET_PATH, '100/feats')
    DATASET_LABELS = os.path.join(DATASET_PATH, '100/labels')
    DATASET_NUMPY = 'data.npz'
    #print(DATASET_PATH)
    if not os.path.isfile(DATASET_FEATURES):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
                                   os.path.join(DATASET_PATH, 'tmp.tgz'))
        print('Dataset Dowloaded')
        #print(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

    if not os.path.isfile(os.path.join(DATASET_PATH, DATASET_NUMPY)):
        print('Creating data.npz file from raw data')
        data_set_features = np.genfromtxt(DATASET_FEATURES, delimiter=',')
        data_set_label = np.genfromtxt(DATASET_LABELS, delimiter=',')
        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32) - 1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)

    ## Load data
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    tensor_data = torch.tensor(data['X'], dtype=torch.float32)
    tensor_labels = torch.tensor(data['Y'], dtype=torch.long)
    dataset = CustomDataset(tensor_data, tensor_labels)
    return dataset


def prepare_purchase(data_root):
    '''
    purchase
    X: (197324, 600) binary feature
    Y: (197324,)  100 classes
    '''
    DATASET_PATH = os.path.join(data_root, 'Purchase')
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    DATASET_NAME = 'dataset_purchase'
    DATASET_NUMPY = 'data.npz'
    DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

    if not os.path.isfile(DATASET_FILE):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
                                   os.path.join(DATASET_PATH, 'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

    if not os.path.isfile(os.path.join(DATASET_PATH, DATASET_NUMPY)):
        print('Creating data.npz file from raw data')
        data_set = np.genfromtxt(DATASET_FILE, delimiter=',')
        X = data_set[:, 1:].astype(np.float64)
        Y = (data_set[:, 0]).astype(np.int32) - 1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)

    ## Load data
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    tensor_data = torch.tensor(data['X'], dtype=torch.float32)
    tensor_labels = torch.tensor(data['Y'], dtype=torch.long)
    dataset = CustomDataset(tensor_data, tensor_labels)
    return dataset
