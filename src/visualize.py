import os
import sys
import numpy
import torch
import torch.nn as nn
from scipy.fftpack import fft
import mne
import numpy as np
from loader import  get_image
from torch.autograd import Variable
from models import BSIExtractorModel, BSIClassifierModel, BSIExtractorModel_Deep_dream
from utils import to_device
from Deap_Dream import deep_dream

class Visualizer():
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.device = self._device()
        self.confidence_threshold = 0.9
        self.count = 0
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._load_weights()
        if self.args.vis_type == 'analysis_activations':
            self.analyse_activations()
        if self.args.vis_type == 'deep_dream':
            self.get_dream()

    def _build_loaders(self):
        self.dataset = get_image(path=self.data["filepath"])

    def _build_model(self):
        if self.args.vis_type == 'deep_dream':
            self.extractor_model: nn.Module = to_device(BSIExtractorModel_Deep_dream(), self.device)
        else:
            self.extractor_model: nn.Module = to_device(BSIExtractorModel(), self.device)
        self.classifier_model: nn.Module = to_device(BSIClassifierModel(), self.device)

    def _device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_weights(self):
        if self.args.vis_type == 'deep_dream':
            checkpoint = torch.load(self.data["model_path_vis_deep_dream"])
        else:
            checkpoint = torch.load(self.data["model_path_vis_activations"])
        self.extractor_model.load_state_dict(checkpoint['extractor_model'])
        self.classifier_model.load_state_dict(checkpoint['classifier_model'])
        self.extractor_model.eval()
        self.classifier_model.eval()

    def get_dream(self):
        self.pretrained_model = self.extractor_model.features
        self.answer = []
        for main_subject in range(36):
            self.get_dream_activations(subj_threshold=10, cnn_layer=11, filter_pos=10, main_subject=main_subject)
        self.deep_dream_results = numpy.mean(self.answer,axis = 0)

    def get_dream_activations(self, subj_threshold, cnn_layer, filter_pos, main_subject):
        count = 0
        max_prob = 0
        for batchidx, [image,label] in enumerate(self.dataset):
            if label.cpu().data.numpy()[0] != main_subject:
                continue
            if count ==subj_threshold:
                break
            eeg_data = image.float().unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            prep_img = Variable(torch.from_numpy(eeg_data.detach().numpy()), requires_grad=True)
            prep_img = to_device(prep_img.float(), self.device)
            eeg_data_tensor = to_device(eeg_data,self.device)
            target_class = to_device(label.long(),self.device)
            embedding = self.extractor_model(eeg_data_tensor)
            labels_pred = self.classifier_model(embedding)
            probs = torch.exp(labels_pred)
            top_p, top_class = probs.topk(1, dim=1)
            prob = top_p.cpu().detach().numpy()[0][0]
            top_class = top_class.cpu().numpy()[0][0]
            if prob < self.confidence_threshold or top_class != main_subject:
                continue
            if prob > 0.99:
                break
        dd  = deep_dream(
            eeg_data_tensor,
            self.pretrained_model,
            iterations=20,
            lr=0.01,
            octave_scale=1,
            num_octaves=10,
        )
        feature_map = dd[0][0]
        Fs = 200
        n_points = feature_map.shape[1]
        frequencies = (Fs/2)*numpy.linspace(0,1,1+int(n_points/2))
        good_matrix_freq=[]
        for channel in range(19):
            timeset = feature_map[channel]
            t = self.getfft(timeset,n_points,frequencies)
            good_matrix_freq.append(numpy.mean(t))
        good_matrix_freq = (good_matrix_freq - numpy.min(good_matrix_freq))/(numpy.max(good_matrix_freq) - numpy.min(good_matrix_freq))
        self.answer.append(good_matrix_freq)


    def analyse_activations(self,threshold=1):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        self.extractor_model.conv1.register_forward_hook(get_activation('conv1'))
        self.answer = []
        for main_subject in range(36):
            self.get_activations(main_subject, threshold, activation)
        self.analysis_activations = numpy.mean(self.answer,axis = 0)

    def get_activations(self, subject, threshold, activation):
        count = 0
        all_filters = []
        for batchidx, [image,label] in enumerate(self.dataset):
            if label.cpu().data.numpy()[0] != subject:
                continue
            if count == threshold:
                break
            eeg_data = image.float().unsqueeze(dim=1)
            eeg_data = eeg_data.contiguous()
            prep_img = Variable(torch.from_numpy(eeg_data.detach().numpy()), requires_grad=True)
            prep_img = to_device(prep_img.float(), self.device)
            eeg_data_tensor = to_device(eeg_data,self.device)
            embedding = self.extractor_model(eeg_data_tensor)
            labels_pred = self.classifier_model(embedding)
            probs = torch.exp(labels_pred)
            top_p, top_class = probs.topk(1, dim=1)
            prob = top_p.cpu().detach().numpy()[0][0]
            top_class = top_class.cpu().numpy()[0][0]
            if prob < self.confidence_threshold or top_class != subject:
                continue
            count+=1
            act = activation['conv1'].squeeze()
            act_numpy = act.cpu().detach().numpy()
            self.answer.append(self.get_feature_maps(act_numpy))

    def get_feature_maps(self, act_numpy):
        feature_maps = []
        for idx in range(20):
            feature_maps.append(act_numpy[idx])
        Fs = 200
        n_points = act_numpy.shape[1]
        frequencies = (Fs/2)*numpy.linspace(0,1,1+int(n_points/2))
        imp = []
        for feature_map in feature_maps:
            good_matrix_freq=[]
            for channel in range(19):
                timeset = feature_map[channel]
                t = self.getfft(timeset,n_points,frequencies)
                good_matrix_freq.append(numpy.mean(t))
            good_matrix_freq = (good_matrix_freq - numpy.min(good_matrix_freq))/(numpy.max(good_matrix_freq) - numpy.min(good_matrix_freq))
            imp.append(good_matrix_freq)
        feature_topo = numpy.mean(imp,axis = 0)
        feature_topo_normalized = (feature_topo - numpy.min(feature_topo))/(numpy.max(feature_topo) - numpy.min(feature_topo))
        return feature_topo_normalized

    def getfft(self,data,n_points,frequencies):
        fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
        return fftdata

    def get_result(self):
        if self.args.vis_type == 'analysis_activations':
            return self.analysis_activations
        if self.args.vis_type == 'deep_dream':
            return self.deep_dream_results
