import torch
import random
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

''' Set device '''
def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return type(x)({key: to_device(val, device) for key, val in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([to_device(item, device) for item in x])
    elif isinstance(x, torch.nn.Module):
        return x.to(device)
    else:
        raise NotImplementedError

''' Set Random Seed '''
def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor
    def __init__( self):
      print("Setting up feature selector for CospCovariance()")

    #Return self nothing else to do here
    def fit( self, X, y = None ):
      return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        ans = np.array([np.mean(elem,axis=-1) for elem in X])
        print(np.any(np.isnan(ans)))
        print(np.all(np.isfinite(ans)))
        # print( np.array([np.mean(elem,axis=-1) for elem in X]).shape)
        return np.array([np.mean(elem,axis=-1) for elem in X])
