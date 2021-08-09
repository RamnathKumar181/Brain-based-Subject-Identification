import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np

class Metrics(nn.Module):
    def __init__(self):
        super(Metrics, self).__init__()
        self.score = {
            "accuracy": 0,
            "accuracy_intruder":0
        }
        self.accuracy = []
        self.accuracy_intruder = []


    def reset(self):
        self.accuracy = []
        self.accuracy_intruder = []
        self.score = {
            "accuracy": 0,
            "accuracy_intruder":0
        }


    def update(self, labels_pred, labels,loss):
        _, y_pred_tags = torch.max(labels_pred, dim = 1)
        labels_tags = labels.long()
        labels_tags = labels_tags.cpu()
        y_pred_tags = y_pred_tags.cpu()
        self.accuracy.append(metrics.accuracy_score(labels_tags.numpy(),y_pred_tags.numpy()))

    def update_intruder(self, labels_pred, labels,loss, intruder_list, confidence_threshold):
        probs = torch.exp(labels_pred)
        top_p, y_pred_tags = probs.topk(1, dim=1)
        labels_tags = labels.long()
        labels_tags = labels_tags.cpu()
        y_pred_tags = y_pred_tags.cpu()
        prob = top_p.cpu().detach().numpy()[0][0]
        if prob<confidence_threshold:
            y_pred_tags[0][0] = 37
        labels_tags = labels_tags.numpy()
        if labels_tags[0] in intruder_list:
            labels_tags[0]= 37
            self.accuracy_intruder.append(metrics.accuracy_score(labels_tags,y_pred_tags.numpy()))
            self.precision_intruder.append(metrics.precision_score(labels_tags,y_pred_tags.numpy(), average='micro'))
            self.recall_intruder.append(metrics.recall_score(labels_tags,y_pred_tags.numpy(), average='micro'))
            self.cohen_kappa_intruder.append(metrics.cohen_kappa_score(labels_tags,y_pred_tags.numpy()))
            self.loss_intruder.append(loss)
        self.accuracy.append(metrics.accuracy_score(labels_tags,y_pred_tags.numpy()))
        self.precision.append(metrics.precision_score(labels_tags,y_pred_tags.numpy(), average='micro'))
        self.recall.append(metrics.recall_score(labels_tags,y_pred_tags.numpy(), average='micro'))
        self.cohen_kappa.append(metrics.cohen_kappa_score(labels_tags,y_pred_tags.numpy()))
        self.loss.append(loss)

    def evaluate(self):
        self.score["accuracy"] = np.mean(self.accuracy) * 100

        self.score["accuracy_intruder"] = np.mean(self.accuracy_intruder) * 100

        return self.score
