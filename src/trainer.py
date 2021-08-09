import os
import sys
import numpy
import pandas as pd
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from loader import  get_dataloader, get_dataset_reimenn, get_dataset_intruder, get_dataset_reimenn_intruder
from models import BSIExtractorModel, BSIClassifierModel, EEGNet_old, EEGNet, ShallowNet, DeepNet
from utils import to_device, FeatureSelector, seed_everything
from metrics import Metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from pyriemann.estimation import Covariances, CospCovariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

"""
BSIT Model
"""
class BSITrainer():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.checkpoint_path = '../configs/'+self.args.exp_name+"_"+self.data["filename"]+'.pth'
        self.device = self._device()
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        seed_everything(self.args.seed)
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        if self.args.intruder:
            self.test()
        else:
            self.train()

    def test(self):
        self.extractor_model.load_state_dict(torch.load(self.checkpoint_path)['extractor_model'])
        self.classifier_model.load_state_dict(torch.load(self.checkpoint_path)['classifier_model'])
        X,y = get_dataset_intruder(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])
        self.extractor_model.eval()
        self.classifier_model.eval()
        final = []
        dataloader = DataLoader(TensorDataset(X, y), batch_size = 100)
        for batchid, data in enumerate(dataloader):
            eeg_data ,labels = data
            eeg_data = to_device(eeg_data.float(), self.device)
            eeg_data = eeg_data.unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            labels = to_device(labels.long(), self.device)
            embedding = self.extractor_model(eeg_data)
            labels_pred = self.classifier_model(embedding)
            y_pred = torch.exp(labels_pred)
            top_p, y_pred_tags = y_pred.topk(1, dim=1)
            labels_tags = labels.long()
            labels_tags = labels_tags.cpu()
            y_pred_tags = y_pred_tags.cpu()
            prob = top_p.cpu().detach().numpy()
            y= labels.cpu().detach().numpy()
            for i in range(len(y)):
                final.append([prob[i][0],y[i]])
        numpy.save(f"../outputs/{self.args.exp_name}_{self.data['filename']}.npy", numpy.asarray(final))
        exit(0)

    def _build_loaders(self):
        self.train_loader,self.test_loader = get_dataloader(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])

    def _build_model(self):
        self.extractor_model: nn.Module = to_device(BSIExtractorModel(), self.device)
        self.classifier_model: nn.Module = to_device(BSIClassifierModel(), self.device)

    def _build_criteria_and_optim(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(list(self.extractor_model.parameters()) + list(self.classifier_model.parameters()), lr=self.args.lr)
        self.metrics = Metrics()

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.current_train = self.train_epoch()
            self.current_test = self.validate_epoch()
            self.checkpoint()
            self.step_scheduler()

    def train_epoch(self):
        self.metrics.reset()
        self.extractor_model.train()
        self.classifier_model.train()
        for batchid, data in enumerate(self.train_loader):
            eeg_data,labels = data
            eeg_data = to_device(eeg_data.float(), self.device)
            eeg_data = eeg_data.unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            labels = to_device(labels.long(), self.device)
            embedding = self.extractor_model(eeg_data)
            labels_pred = self.classifier_model(embedding)
            loss = self.criterion(labels_pred, labels.long())
            self.metrics.update(labels_pred,labels.long(),loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def validate_epoch(self):
        self.metrics.reset()
        self.extractor_model.eval()
        self.classifier_model.eval()
        for batchid, data in enumerate(self.test_loader):
            eeg_data_test, labels_test = data
            eeg_data_test = to_device(eeg_data_test.float(),self.device)
            eeg_data_test = eeg_data_test.unsqueeze(dim=1)
            eeg_data_test = eeg_data_test.contiguous()
            labels_test = to_device(labels_test.long(), self.device)
            embedding = self.extractor_model(eeg_data_test)
            labels_pred = self.classifier_model(embedding)
            loss = self.criterion(labels_pred, labels_test.long())
            self.metrics.update(labels_pred,labels_test.long(),loss.item())
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def checkpoint(self, force=True):
        if self.highest_test < self.current_test:
            self.highest_test = self.current_test
            self.highest_train = self.current_train
            info_dict = {
                "extractor_model": self.extractor_model.state_dict(),
                "classifier_model": self.classifier_model.state_dict(),
                "optim": self.optimizer.state_dict(),
            }
            torch.save(info_dict, self.checkpoint_path)
        return self

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        return self

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

"""
EEGNet Model
"""
class EEGNetTrainer():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.checkpoint_path = '../configs/'+self.args.exp_name+"_"+self.data["filename"]+'.pth'
        self.device = self._device()
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        self.train()

    def _build_loaders(self):
        self.train_loader,self.test_loader = get_dataloader(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])

    def _build_model(self):
        self.eegnet_model: nn.Module = to_device(EEGNet(), self.device)

    def _build_criteria_and_optim(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.eegnet_model.parameters(), lr=self.args.lr)
        self.metrics = Metrics()

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.current_train = self.train_epoch()
            self.current_test = self.validate_epoch()
            self.checkpoint()
            self.step_scheduler()

    def train_epoch(self):
        self.metrics.reset()
        self.eegnet_model.train()
        for batchid, data in enumerate(self.train_loader):
            eeg_data,labels = data
            eeg_data = to_device(eeg_data.float(), self.device)
            eeg_data = eeg_data.unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            labels = to_device(labels.long(), self.device)
            labels_pred = self.eegnet_model(eeg_data)
            loss = self.criterion(labels_pred, labels.long())
            self.metrics.update(labels_pred,labels.long(),loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def validate_epoch(self):
        self.metrics.reset()
        self.eegnet_model.eval()
        for batchid, data in enumerate(self.test_loader):
            eeg_data_test, labels_test = data
            eeg_data_test = to_device(eeg_data_test.float(),self.device)
            eeg_data_test = eeg_data_test.unsqueeze(dim=1)
            eeg_data_test = eeg_data_test.contiguous()
            labels_test = to_device(labels_test.long(), self.device)
            labels_pred = self.eegnet_model(eeg_data_test)
            loss = self.criterion(labels_pred, labels_test.long())
            self.metrics.update(labels_pred,labels_test.long(),loss.item())
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def checkpoint(self, force=True):
        if self.highest_test < self.current_test:
            self.highest_test = self.current_test
            self.highest_train = self.current_train
            info_dict = {
                "eegnet_model": self.eegnet_model.state_dict(),
                "optim": self.optimizer.state_dict(),
            }
            torch.save(info_dict, self.checkpoint_path)
        return self

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        return self

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

"""
EEGNet_old Model
"""
class EEGNet_old_Trainer():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.checkpoint_path = '../configs/'+self.args.exp_name+"_"+self.data["filename"]+'.pth'
        self.device = self._device()
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        self.train()

    def _build_loaders(self):
        self.train_loader,self.test_loader = get_dataloader(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])

    def _build_model(self):
        self.eegnet_model: nn.Module = to_device(EEGNet_old(), self.device)

    def _build_criteria_and_optim(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.eegnet_model.parameters(), lr=self.args.lr)
        self.metrics = Metrics()

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.current_train = self.train_epoch()
            self.current_test = self.validate_epoch()
            self.checkpoint()
            self.step_scheduler()

    def train_epoch(self):
        self.metrics.reset()
        self.eegnet_model.train()
        for batchid, data in enumerate(self.train_loader):
            eeg_data,labels = data
            eeg_data = to_device(eeg_data.float(), self.device)
            eeg_data = eeg_data.unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            labels = to_device(labels.long(), self.device)
            labels_pred = self.eegnet_model(eeg_data)
            loss = self.criterion(labels_pred, labels.long())
            self.metrics.update(labels_pred,labels.long(),loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def validate_epoch(self):
        self.metrics.reset()
        self.eegnet_model.eval()
        for batchid, data in enumerate(self.test_loader):
            eeg_data_test, labels_test = data
            eeg_data_test = to_device(eeg_data_test.float(),self.device)
            eeg_data_test = eeg_data_test.unsqueeze(dim=1)
            eeg_data_test = eeg_data_test.contiguous()
            labels_test = to_device(labels_test.long(), self.device)
            labels_pred = self.eegnet_model(eeg_data_test)
            loss = self.criterion(labels_pred, labels_test.long())
            self.metrics.update(labels_pred,labels_test.long(),loss.item())
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def checkpoint(self, force=True):
        if self.highest_test < self.current_test:
            self.highest_test = self.current_test
            self.highest_train = self.current_train
            info_dict = {
                "eegnet_model": self.eegnet_model.state_dict(),
                "optim": self.optimizer.state_dict(),
            }
            torch.save(info_dict, self.checkpoint_path)
        return self

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        return self

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

"""
ShallowNet Model
"""
class ShallowNetTrainer():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.checkpoint_path = '../configs/'+self.args.exp_name+"_"+self.data["filename"]+'.pth'
        self.device = self._device()
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        self.train()

    def _build_loaders(self):
        self.train_loader,self.test_loader = get_dataloader(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])

    def _build_model(self):
        self.shallownet_model: nn.Module = to_device(ShallowNet(), self.device)

    def _build_criteria_and_optim(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.shallownet_model.parameters(), lr=self.args.lr)
        self.metrics = Metrics()

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.current_train = self.train_epoch()
            self.current_test = self.validate_epoch()
            self.checkpoint()
            self.step_scheduler()

    def train_epoch(self):
        self.metrics.reset()
        self.shallownet_model.train()
        for batchid, data in enumerate(self.train_loader):
            eeg_data,labels = data
            eeg_data = to_device(eeg_data.float(), self.device)
            eeg_data = eeg_data.unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            labels = to_device(labels.long(), self.device)
            labels_pred = self.shallownet_model(eeg_data)
            loss = self.criterion(labels_pred, labels.long())
            self.metrics.update(labels_pred,labels.long(),loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def validate_epoch(self):
        self.metrics.reset()
        self.shallownet_model.eval()
        for batchid, data in enumerate(self.test_loader):
            eeg_data_test, labels_test = data
            eeg_data_test = to_device(eeg_data_test.float(),self.device)
            eeg_data_test = eeg_data_test.unsqueeze(dim=1)
            eeg_data_test = eeg_data_test.contiguous()
            labels_test = to_device(labels_test.long(), self.device)
            labels_pred = self.shallownet_model(eeg_data_test)
            loss = self.criterion(labels_pred, labels_test.long())
            self.metrics.update(labels_pred,labels_test.long(),loss.item())
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def checkpoint(self, force=True):
        if self.highest_test < self.current_test:
            self.highest_test = self.current_test
            self.highest_train = self.current_train
            info_dict = {
                "shallownet_model": self.shallownet_model.state_dict(),
                "optim": self.optimizer.state_dict(),
            }
            torch.save(info_dict, self.checkpoint_path)
        return self

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        return self

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

"""
DeepNet Model
"""
class DeepNetTrainer():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.checkpoint_path = '../configs/'+self.args.exp_name+"_"+self.data["filename"]+'.pth'
        self.device = self._device()
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()
        self.train()

    def _build_loaders(self):
        self.train_loader,self.test_loader = get_dataloader(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])

    def _build_model(self):
        self.deepnet_model: nn.Module = to_device(DeepNet(), self.device)

    def _build_criteria_and_optim(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.deepnet_model.parameters(), lr=self.args.lr)
        self.metrics = Metrics()

    def _build_scheduler(self):
        rate_decay_step_size = 40
        rate_decay_gamma = 0.8
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=rate_decay_step_size, gamma=rate_decay_gamma)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.current_train = self.train_epoch()
            self.current_test = self.validate_epoch()
            self.checkpoint()
            self.step_scheduler()

    def train_epoch(self):
        self.metrics.reset()
        self.deepnet_model.train()
        for batchid, data in enumerate(self.train_loader):
            eeg_data,labels = data
            eeg_data = to_device(eeg_data.float(), self.device)
            eeg_data = eeg_data.unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            labels = to_device(labels.long(), self.device)
            labels_pred = self.deepnet_model(eeg_data)
            loss = self.criterion(labels_pred, labels.long())
            self.metrics.update(labels_pred,labels.long(),loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def validate_epoch(self):
        self.metrics.reset()
        self.deepnet_model.eval()
        for batchid, data in enumerate(self.test_loader):
            eeg_data_test, labels_test = data
            eeg_data_test = to_device(eeg_data_test.float(),self.device)
            eeg_data_test = eeg_data_test.unsqueeze(dim=1)
            eeg_data_test = eeg_data_test.contiguous()
            labels_test = to_device(labels_test.long(), self.device)
            labels_pred = self.deepnet_model(eeg_data_test)
            loss = self.criterion(labels_pred, labels_test.long())
            self.metrics.update(labels_pred,labels_test.long(),loss.item())
        all_metrics = self.metrics.evaluate()
        return all_metrics["accuracy"]

    def checkpoint(self, force=True):
        if self.highest_test < self.current_test:
            self.highest_test = self.current_test
            self.highest_train = self.current_train
            info_dict = {
                "deepnet_model": self.deepnet_model.state_dict(),
                "optim": self.optimizer.state_dict(),
            }
            torch.save(info_dict, self.checkpoint_path)
        return self

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
        return self

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

"""
KNN Model
"""
class KNNTrainer_riemann():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self.train()

    def _build_loaders(self):
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_reimenn(path=self.data["filepath"],
                                                                                  test_split=self.args.split_ratio,
                                                                                  batch_size=self.args.batch_size,
                                                                                  threshold=self.data["threshold"],
                                                                                  preprocess=self.data["pre"],
                                                                                  lf=self.data["lf"],
                                                                                  hf=self.data["hf"])

    def _build_model(self):
        knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', metric = 'minkowski', p = 2)
        if self.args.riemann_type == 'time':
            self.clf = make_pipeline(Covariances('lwf'),
                                     TangentSpace(metric='riemann'),
                                     knn)
        elif self.args.riemann_type == 'frequency':
            self.clf = make_pipeline(CospCovariances(),
                                     FeatureSelector(),
                                     TangentSpace(metric='riemann'),
                                     knn)
        else:
            self.clf = make_pipeline(Covariances('lwf'),
                                     CSP(8, 'riemann', False),
                                     TangentSpace(metric='riemann'),
                                     knn)

    def train(self):
        self.clf.fit(numpy.array(self.X_train), self.y_train)
        y_pred_train = self.clf.predict(numpy.array(self.X_train))
        y_pred = self.clf.predict(numpy.array(self.X_test))

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100
        self.cm = confusion_matrix(self.y_test, y_pred)

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

    def get_confusion_matrix(self):
        return self.cm

"""
RF Model
"""
class RFTrainer_riemann():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self.train()

    def _build_loaders(self):
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_reimenn(path=self.data["filepath"],
                                                                                  test_split=self.args.split_ratio,
                                                                                  batch_size=self.args.batch_size,
                                                                                  threshold=self.data["threshold"],
                                                                                  preprocess=self.data["pre"],
                                                                                  lf=self.data["lf"],
                                                                                  hf=self.data["hf"])

    def _build_model(self):
        rf = RandomForestClassifier(n_estimators = 170, criterion = 'entropy')
        if self.args.riemann_type == 'time':
            self.clf = make_pipeline(Covariances('lwf'),
                                     TangentSpace(metric='riemann'),
                                     rf)
        elif self.args.riemann_type == 'frequency':
            self.clf = make_pipeline(CospCovariances(),
                                     FeatureSelector(),
                                     TangentSpace(metric='riemann'),
                                     rf)
        else:
            self.clf = make_pipeline(Covariances('lwf'),
                                     CSP(8, 'riemann', False),
                                     TangentSpace(metric='riemann'),
                                     rf)

    def train(self):
        self.clf.fit(numpy.array(self.X_train), self.y_train)
        y_pred_train = self.clf.predict(numpy.array(self.X_train))
        y_pred = self.clf.predict(numpy.array(self.X_test))

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100
        self.cm = confusion_matrix(self.y_test, y_pred)

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

    def get_confusion_matrix(self):
        return self.cm

"""
GB Model
"""
class GBTrainer_riemann():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self.train()

    def _build_loaders(self):
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_reimenn(path=self.data["filepath"],
                                                                                  test_split=self.args.split_ratio,
                                                                                  batch_size=self.args.batch_size,
                                                                                  threshold=self.data["threshold"],
                                                                                  preprocess=self.data["pre"],
                                                                                  lf=self.data["lf"],
                                                                                  hf=self.data["hf"])

    def _build_model(self):
        gb = GradientBoostingClassifier(n_estimators = 150)
        if self.args.riemann_type == 'time':
            self.clf = make_pipeline(Covariances('lwf'),
                                     TangentSpace(metric='riemann'),
                                     gb)
        elif self.args.riemann_type == 'frequency':
            self.clf = make_pipeline(CospCovariances(),
                                     FeatureSelector(),
                                     TangentSpace(metric='riemann'),
                                     gb)
        else:
            self.clf = make_pipeline(Covariances('lwf'),
                                     CSP(8, 'riemann', False),
                                     TangentSpace(metric='riemann'),
                                     gb)

    def train(self):
        self.clf.fit(numpy.array(self.X_train), self.y_train)
        y_pred_train = self.clf.predict(numpy.array(self.X_train))
        y_pred = self.clf.predict(numpy.array(self.X_test))

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100
        self.cm = confusion_matrix(self.y_test, y_pred)

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

    def get_confusion_matrix(self):
        return self.cm

"""
SVM Model
"""
class SVMTrainer_riemann():

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.highest_train = 0
        self.highest_test = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self.train()
        if self.args.intruder:
            self.test()

    def test(self):
        X,y = get_dataset_reimenn_intruder(path=self.data["filepath"], test_split=self.args.split_ratio, batch_size=self.args.batch_size, threshold=self.data["threshold"], preprocess=self.data["pre"], lf=self.data["lf"], hf=self.data["hf"])
        y_pred = self.clf.predict_proba(numpy.array(X))
        final = []
        prob = numpy.amax(y_pred,axis=1)
        for i in range(len(y)):
            final.append([prob[i],y[i]])
        numpy.save(f"../outputs/{self.args.exp_name}_{self.data['filename']}.npy", numpy.asarray(final))
        exit(0)

    def _build_loaders(self):
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_reimenn(path=self.data["filepath"],
                                                                                  test_split=self.args.split_ratio,
                                                                                  batch_size=self.args.batch_size,
                                                                                  threshold=self.data["threshold"],
                                                                                  preprocess=self.data["pre"],
                                                                                  lf=self.data["lf"],
                                                                                  hf=self.data["hf"])

    def _build_model(self):
        svm_clf = svm.SVC(kernel=self.args.svm_kernel, probability=True)
        if self.args.riemann_type == 'time':
            self.clf = make_pipeline(Covariances('lwf'),
                                     TangentSpace(metric='riemann'),
                                     svm_clf)
        elif self.args.riemann_type == 'frequency':
            self.clf = make_pipeline(CospCovariances(),
                                     FeatureSelector(),
                                     TangentSpace(metric='riemann'),
                                     svm_clf)
        else:
            self.clf = make_pipeline(Covariances('lwf'),
                                     CSP(8, 'riemann', False),
                                     TangentSpace(metric='riemann'),
                                     svm_clf)

    def train(self):
        self.clf.fit(numpy.array(self.X_train), self.y_train)
        y_pred_train = self.clf.predict(numpy.array(self.X_train))
        y_pred = self.clf.predict(numpy.array(self.X_test))

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100
        self.cm = confusion_matrix(self.y_test, y_pred)

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])

    def get_confusion_matrix(self):
        return self.cm
