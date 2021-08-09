import os
import sys
import numpy
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from loader import  get_dataloader, get_dataset_handcrafted_features
from models import BSIExtractorModel, BSIClassifierModel, EEGNet_old, EEGNet, ShallowNet, DeepNet
from utils import to_device
from metrics import Metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


"""
SVM Model
"""
class SVMTrainer():

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
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_handcrafted_features(path=self.data["filepath"],
                                                                                               test_split=self.args.split_ratio,
                                                                                               batch_size=self.args.batch_size,
                                                                                               threshold=self.data["threshold"],
                                                                                               preprocess=self.data["pre"],
                                                                                               lf=self.data["lf"],
                                                                                               hf=self.data["hf"],
                                                                                               channel = self.data["channel"])

    def _build_model(self):
        self.clf = svm.SVC(kernel=self.args.svm_kernel)

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
        y_pred_train = self.clf.predict(self.X_train)
        y_pred = self.clf.predict(self.X_test)

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])


"""
KNN Model
"""
class KNNTrainer_Reimenn():

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
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_handcrafted_features(path=self.data["filepath"],
                                                                                               test_split=self.args.split_ratio,
                                                                                               batch_size=self.args.batch_size,
                                                                                               threshold=self.data["threshold"],
                                                                                               preprocess=self.data["pre"],
                                                                                               lf=self.data["lf"],
                                                                                               hf=self.data["hf"],
                                                                                               channel = self.data["channel"])

    def _build_model(self):
        self.clf = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', metric = 'minkowski', p = 2)

    def train(self):

        self.clf.fit(self.X_train, self.y_train)
        y_pred_train = self.clf.predict(self.X_train)
        y_pred = self.clf.predict(self.X_test)

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])


"""
Random Forest Model
"""
class RFTrainer():

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
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_handcrafted_features(path=self.data["filepath"],
                                                                                               test_split=self.args.split_ratio,
                                                                                               batch_size=self.args.batch_size,
                                                                                               threshold=self.data["threshold"],
                                                                                               preprocess=self.data["pre"],
                                                                                               lf=self.data["lf"],
                                                                                               hf=self.data["hf"],
                                                                                               channel = self.data["channel"])
    def _build_model(self):
        self.clf = RandomForestClassifier(n_estimators = 170, criterion = 'entropy')


    def train(self):
        self.clf.fit(self.X_train, self.y_train)
        y_pred_train = self.clf.predict(self.X_train)
        y_pred = self.clf.predict(self.X_test)

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(self.y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(self.y_test,y_pred) * 100

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])


"""
Gradient Boosting Model
"""
class GBTrainer():

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
        self.X_train, self.X_test, self.y_train, self.y_test= get_dataset_handcrafted_features(path=self.data["filepath"],
                                                                                               test_split=self.args.split_ratio,
                                                                                               batch_size=self.args.batch_size,
                                                                                               threshold=self.data["threshold"],
                                                                                               preprocess=self.data["pre"],
                                                                                               lf=self.data["lf"],
                                                                                               hf=self.data["hf"],
                                                                                               channel = self.data["channel"])
    def _build_model(self):
        self.clf = GradientBoostingClassifier(n_estimators = 150)

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
        y_pred_train = classifier.predict(X_train)
        y_pred = classifier.predict(X_test)

        from sklearn import metrics
        self.highest_train = metrics.accuracy_score(y_train,y_pred_train) * 100
        self.highest_test = metrics.accuracy_score(y_test,y_pred) * 100

    def get_result(self):
        return tuple([self.highest_train, self.highest_test])
