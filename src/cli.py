"""Console script for project Brain based subject identification."""
import os
from argparse import ArgumentParser

def parse_args():
    """
    Parse arguments
    """
    parser = ArgumentParser(
        description = "Basic Interface for the Brain-based subject identification"
    )
    parser.add_argument("--dataset",
                        required=True,
                        type=int,
                        choices=[0,1,2,3,4,5],
                        help="Dataset to be used. Options include %(choices)")
    parser.add_argument("--exp_name",
                        required=False,
                        type=str,
                        default='exp',
                        help="Name of experiment (Default = 'exp')")
    parser.add_argument("--model",
                        required=False,
                        type=str,
                        choices=['bsit','svm','knn','rf','gb', 'eegnet_old', 'eegnet', 'shallow_net', 'deep_net'],
                        help="Model to be used. Options include %(choices)")
    parser.add_argument("--lr",
                        required=False,
                        type=float,
                        default=0.0008,
                        help="Learning rate (Default = 0.0008)")
    parser.add_argument("--epochs",
                        required=False,
                        type=int,
                        default=100,
                        help="Number of epochs (Default = 100)")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=32,
                        help="Batch size (Default = 32)")
    parser.add_argument("--runs",
                        required=False,
                        type=int,
                        default=5,
                        help="Number of runs (Default = 5)")
    parser.add_argument("--split_ratio",
                        required=False,
                        type=float,
                        default=0.2,
                        help="Train-test split ratio (Default = 0.2)")
    parser.add_argument("--eval",
                        required=False,
                        default=False,
                        dest="eval",
                        action="store_true",
                        help="Set to true if you want to evaluate your model")
    parser.add_argument("--frequency_bands",
                        required=False,
                        default='all',
                        type=str,
                        choices=['all','alpha','beta','theta','delta'],
                        help="Frequency band to be used. Options include %(choices)")
    parser.add_argument("--svm_kernel",
                        required=False,
                        default='rbf',
                        type=str,
                        choices=['linear', 'rbf'],
                        help="Model to be used. Options include %(choices)")
    parser.add_argument("--vis",
                        required=False,
                        default=False,
                        dest="vis",
                        action="store_true",
                        help="Set to true if you want to visualize your model results")
    parser.add_argument("--vis_type",
                        required=False,
                        type=str,
                        choices=['analysis_weights','analysis_activations','deep_dream','guided_backprop'],
                        help="Model to be used. Options include %(choices)")
    parser.add_argument("--channel_level",
                        required=False,
                        default=False,
                        dest="channel_level",
                        action="store_true",
                        help="Set to true if you want to run per electrode")
    parser.add_argument("--plot_confusion_matrix",
                        required=False,
                        default=False,
                        dest="plot_confusion_matrix",
                        action="store_true",
                        help="Set to true if you want to plot confusion matrix")
    parser.add_argument("--minimal_data_needed",
                        required=False,
                        default=False,
                        dest="minimal_data_needed",
                        action="store_true",
                        help="Set to true if you want to compute performance for different train test split ratios")
    parser.add_argument("--intruder",
                        required=False,
                        default=False,
                        dest="intruder",
                        action="store_true",
                        help="Set to true if you want to compute intruder detection")
    parser.add_argument("--absolute_values",
                        required=False,
                        type=int,
                        default=0,
                        help="Need to use absolute runs or not (Default = 0, Use 1 to use absolute values)")
    parser.add_argument("--seed",
                        required=False,
                        type=int,
                        default=0,
                        help="Set seed")
    parser.add_argument('--model_path',
                        type=str,
                        default='',
                        help="Pretrained model path (Default = '')")
    parser.add_argument("--riemann_type",
                        required=False,
                        type=str,
                        choices=['time','frequency', 'csp'],
                        help="riemann Covariance to be used. Options include %(choices)")
    args = parser.parse_args()
    return args
