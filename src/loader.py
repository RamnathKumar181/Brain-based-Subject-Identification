import os
import io
import h5py
import numpy as np
import torch
import mne
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import random
from mne.time_frequency import psd_welch
import scipy.stats
import math


class Brain_Concat_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        return tuple([self.dataset1[i] , self.dataset2[i]-1])

    def __len__(self):
        return len(self.dataset1)


class BrainDataset(Dataset):

    def __init__(self, path, lf,hf,preprocess):
        self.file = h5py.File(path,'r')
        self.data = self.file["home"]["data"]
        self.label = self.file["home"]["subj_labels"]
        self.lf = lf
        self.hf = hf
        if preprocess:
            self.new_data = self.preprocess(self.data)
        else:
            self.new_data = self.data
        self.total_dataset = Brain_Concat_Dataset(self.new_data , self.label)

    def __getitem__(self, i):
        return self.total_dataset[i]

    def preprocess(self,data):
        ndata = torch.zeros(data.shape)
        for i in range(data.shape[0]):
            ndata[i] = torch.tensor(mne.filter.filter_data(data[i].T,200, l_freq=self.lf, h_freq=self.hf, picks=None, filter_length='auto',h_trans_bandwidth=2,verbose=False).T)
        return ndata

    def __len__(self):
        return len(self.total_dataset)


def split_dataset(dataset, test_split,total_count,threshold):
    X = []
    y = []
    current_count = {}
    for i in range(36):
        current_count[i]=0
    for i in range(len(dataset)):
        if(total_count[dataset[i][1]]>=threshold and current_count[dataset[i][1]]<threshold):
            X.append(dataset[i][0])
            y.append(dataset[i][1]+1)
            current_count[dataset[i][1]]+=1
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=test_split)
    return X_train, X_test, y_train, y_test


def split_dataset_reimenn(dataset, test_split,total_count,threshold):
    X = []
    y = []
    current_count = {}
    for i in range(36):
        current_count[i]=0
    for i in range(len(dataset)):
        if(total_count[dataset[i][1]]>=threshold and current_count[dataset[i][1]]<threshold):
            if isinstance(dataset[i][0], np.ndarray):
                X.append(dataset[i][0].T)
            else:
                X.append(dataset[i][0].detach().cpu().numpy().T)
            y.append(dataset[i][1]+1)
            current_count[dataset[i][1]]+=1
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=test_split)
    return X_train, X_test, y_train, y_test

def get_dataloader(path, batch_size, test_split, threshold, lf, hf, preprocess, shuffle=True,**dataset_kwargs):
    X_train, X_test, y_train, y_test =  get_dataset(path, batch_size, test_split, threshold, lf, hf, preprocess)
    train_dataset = Brain_Concat_Dataset(X_train , y_train)
    test_dataset = Brain_Concat_Dataset(X_test , y_test)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_dataloader,test_dataloader


def split_dataset_intruder(dataset, test_split,total_count,threshold):
    X = []
    y = []
    for i in range(len(dataset)):
        if(total_count[dataset[i][1]]<threshold):
            X.append(dataset[i][0])
            y.append(1)
    num_intruders = len(X)
    count=0
    for i in range(len(dataset)):
        if(total_count[dataset[i][1]]>=threshold and count<num_intruders):
            X.append(dataset[i][0])
            y.append(0)
            count+=1
    return X,y


def get_dataset_intruder(path, batch_size, test_split, threshold, lf, hf, preprocess, shuffle=True,**dataset_kwargs):

    if isinstance(path, str):
        dataset = BrainDataset(path=path,lf = lf,hf = hf,preprocess = preprocess)
    elif isinstance(path, (list, tuple)):
        dataset = ConcatDataset(
            [BrainDataset(path=p,lf=lf,hf=hf,preprocess = preprocess) for p in path]
        )
    else:
        raise TypeError
    total_count = {}
    for i in range(36):
        total_count[i]=0
    for i in range(len(dataset)):
        total_count[dataset[i][1]] +=1
    X,y = split_dataset_intruder(dataset,test_split,total_count,threshold)
    return torch.FloatTensor(X),torch.FloatTensor(y)

def get_dataset(path, batch_size, test_split, threshold, lf, hf, preprocess, shuffle=True,**dataset_kwargs):

    if isinstance(path, str):
        dataset = BrainDataset(path=path,lf = lf,hf = hf,preprocess = preprocess)
    elif isinstance(path, (list, tuple)):
        dataset = ConcatDataset(
            [BrainDataset(path=p,lf=lf,hf=hf,preprocess = preprocess) for p in path]
        )
    else:
        raise TypeError
    total_count = {}
    for i in range(36):
        total_count[i]=0
    for i in range(len(dataset)):
        total_count[dataset[i][1]] +=1
    X_train, X_test, y_train, y_test = split_dataset(dataset,test_split,total_count,threshold)
    return X_train, X_test, y_train, y_test

def get_image_dataset(path,**dataset_kwargs):

    if isinstance(path, str):
        dataset = BrainDataset(path=path,lf = 0,hf = 45,preprocess = True)
    elif isinstance(path, (list, tuple)):
        dataset = ConcatDataset(
            [BrainDataset(path=p,lf=0,hf=45,preprocess = True) for p in path]
        )
    else:
        raise TypeError

    return dataset

def get_image(path,**dataset_kwargs):

    dataset = get_image_dataset(path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )
    return dataloader

def get_dataset_reimenn(path, batch_size, test_split, threshold, lf, hf, preprocess, shuffle=True,**dataset_kwargs):

    if isinstance(path, str):
        dataset = BrainDataset(path=path,lf = lf,hf = hf,preprocess = preprocess)
    elif isinstance(path, (list, tuple)):
        dataset = ConcatDataset(
            [BrainDataset(path=p,lf=lf,hf=hf,preprocess = preprocess) for p in path]
        )
    else:
        raise TypeError
    total_count = {}
    for i in range(36):
        total_count[i]=0
    for i in range(len(dataset)):
        total_count[dataset[i][1]] +=1
    X_train, X_test, y_train, y_test = split_dataset_reimenn(dataset,test_split,total_count,threshold)
    return X_train, X_test, y_train, y_test

def split_dataset_reimenn_intruder(dataset, test_split,total_count,threshold):
    X = []
    y = []
    for i in range(len(dataset)):
        if(total_count[dataset[i][1]]<threshold):
            if isinstance(dataset[i][0], np.ndarray):
                X.append(dataset[i][0].T)
            else:
                X.append(dataset[i][0].detach().cpu().numpy().T)
            y.append(1)
    num_intruders = len(X)
    count=0
    for i in range(len(dataset)):
        if(total_count[dataset[i][1]]>=threshold and count<num_intruders):
            if isinstance(dataset[i][0], np.ndarray):
                X.append(dataset[i][0].T)
            else:
                X.append(dataset[i][0].detach().cpu().numpy().T)
            y.append(0)
            count+=1
    return X,y

def get_dataset_reimenn_intruder(path, batch_size, test_split, threshold, lf, hf, preprocess, shuffle=True,**dataset_kwargs):

    if isinstance(path, str):
        dataset = BrainDataset(path=path,lf = lf,hf = hf,preprocess = preprocess)
    elif isinstance(path, (list, tuple)):
        dataset = ConcatDataset(
            [BrainDataset(path=p,lf=lf,hf=hf,preprocess = preprocess) for p in path]
        )
    else:
        raise TypeError
    total_count = {}
    for i in range(36):
        total_count[i]=0
    for i in range(len(dataset)):
        total_count[dataset[i][1]] +=1
    X,y = split_dataset_reimenn_intruder(dataset,test_split,total_count,threshold)
    return X,y
