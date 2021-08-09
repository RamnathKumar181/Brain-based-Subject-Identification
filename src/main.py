"""Console script for project Brain based subject identification."""
import os
from cli import parse_args
from utils import seed_everything
from logger import Logger, Logger_Channel
import numpy
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

elec_names={
0:'Fz',
1:'Cz',
2:'Pz',
3:'C3',
4:'C4',
5:'T3',
6:'T4',
7:'Fp1',
8:'Fp2',
9:'O1',
10:'O2',
11:'F3',
12:'F4',
13:'P3',
14:'P4',
15:'FC1',
16:'FC2',
17:'CP1',
18:'CP2'}
CHANNEL_NAMES=['Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4', 'Fp1', 'Fp2', 'O1','O2', 'F3', 'F4', 'P3', 'P4', 'FC1', 'FC2', 'CP1', 'CP2']
SENSORS_POS=numpy.array([
[0.714,0,0.7],
[6.12e-17,0,1],
[-0.714,-8.74e-17,0.7],
[4.55e-17,0.744,0.668],
[4.55e-17,-0.744,0.668],
[6.09e-17,0.995,-0.103],
[6.09e-17,-0.995,-0.103],
[0.95,0.309,-0.0471],
[0.95,-0.309,-0.0471],
[-0.95,0.309,-0.0471],
[-0.95,-0.309,-0.0471],
[0.677,0.568,0.468],
[0.676,-0.567,0.471],
[-0.677,0.568,0.468],
[-0.676,-0.567,0.471],
[0.381,0.381,0.843],
[0.381,-0.381,0.843],
[-0.381,0.381,0.843],
[-0.381,-0.381,0.843]])[:,:2]
t=SENSORS_POS
t1=numpy.array(t[:,0])
t[:,0]=t[:,1]
t[:,1]=t1
t[:,0] *= -1
SENSORS_POS=t


def main():
    """
    Console script for project Brain based subject identification.
    """
    args = parse_args()
    data = {}
    if args.dataset ==0:
        data = {
        "filepath": "../../ori_sleep_data/all_AWA.h5",
        "threshold": 210.0,
        "filename": "awa",
        "model_path_vis_activations": "../configs/balanced_preprocessing_0_45_AWA.pth",
        "model_path_vis_deep_dream": "../configs/visualiazation_0_45_all_AWA.pth"
        }
    elif args.dataset ==1:
        data = {
        "filepath": "../../ori_sleep_data/all_Rem.h5",
        "threshold": 744.0,
        "filename": "rem",
        "model_path_vis_activations": "../configs/balanced_preprocessing_0_45_Rem.pth",
        "model_path_vis_deep_dream": "../configs/visualiazation_0_45_all_Rem.pth"
        }
    elif args.dataset ==2:
        data = {
        "filepath":  "../../ori_sleep_data/all_S1.h5",
        "threshold": 222.0,
        "filename": "s1",
        "model_path_vis_activations": "../configs/balanced_preprocessing_0_45_S1.pth",
        "model_path_vis_deep_dream": "../configs/visualiazation_0_45_all_S1.pth"
        }
    elif args.dataset ==3:
        data = {
        "filepath":  "../../ori_sleep_data/all_S2.h5",
        "threshold": 1512.0,
        "filename": "s2",
        "model_path_vis_activations": "../configs/balanced_preprocessing_0_45_S2.pth",
        "model_path_vis_deep_dream": "../configs/visualiazation_0_45_all_S2.pth"
        }
    elif args.dataset ==4:
        data = {
        "filepath":  "../../ori_sleep_data/all_SWS.h5",
        "threshold": 1074.0,
        "filename":"sws",
        "model_path_vis_activations": "../configs/balanced_preprocessing_0_45_SWS.pth",
        "model_path_vis_deep_dream": "../configs/visualiazation_0_45_all_SWS.pth"
        }

    """
    Frequency bands devided based on the:
    Recommendations for the Practice of Clinical Neurophysiology: Guidelines of the International Federation of
    Clinical Physiology (EEG Suppl. 52) Editors: G. Deuschl and A. Eisen q 1999 International Federation of
    Clinical Neurophysiology. All rights reserved. Published by Elsevier Science B.V.
    """
    if args.frequency_bands == "alpha":
        data["lf"] = 8
        data["hf"] = 14
        data["pre"] = True
    elif args.frequency_bands == "beta":
        data["lf"] = 14
        data["hf"] = 45
        data["pre"] = True
    elif args.frequency_bands == "delta":
        data["lf"] = 0
        data["hf"] = 4
        data["pre"] = True
    elif args.frequency_bands == "theta":
        data["lf"] = 4
        data["hf"] = 8
        data["pre"] = True
    else:
        data["pre"] = False
        data["lf"] = 0
        data["hf"] = 0

    """
    Training model and baselines
    """
    if not (args.eval or args.vis):
        """
        Train a model
        """
        if args.minimal_data_needed:
            """
            Compute minimal data needed for robust performances
            """
            if not args.absolute_values:
                for split_ratio in numpy.arange(0.1,1.0,0.1):
                    args.split_ratio = split_ratio
                    print(args.split_ratio)
                    logger = Logger(args.runs, args)
                    for run in range(args.runs):
                        import gc
                        gc.collect()
                        seed_everything(run)
                        if args.model == "bsit":
                            """
                            Brain Subject Identification Trainer
                            """
                            from trainer import BSITrainer
                            bt = BSITrainer(args, data)
                            logger.add_result(run, bt.get_result())
                        elif args.model == "svm":
                            """
                             SVM riemann
                            """
                            from trainer import SVMTrainer_riemann
                            st = SVMTrainer_riemann(args, data)
                            logger.add_result(run, st.get_result())
                    logger.print_statistics()
            else:
                for index in [189,168,147,126,105,84,63,42,21]:
                    args.split_ratio = (data["threshold"]-index)/data["threshold"]
                    print(index)
                    logger = Logger(args.runs, args)
                    for run in range(args.runs):
                        import gc
                        gc.collect()
                        seed_everything(run)
                        if args.model == "bsit":
                            """
                            Brain Subject Identification Trainer
                            """
                            from trainer import BSITrainer
                            bt = BSITrainer(args, data)
                            logger.add_result(run, bt.get_result())
                        elif args.model == "svm":
                            """
                             SVM riemann
                            """
                            from trainer import SVMTrainer_riemann
                            st = SVMTrainer_riemann(args, data)
                            logger.add_result(run, st.get_result())
                    logger.print_statistics()

        else:
            if args.channel_level:
                logger_main = Logger(args.runs, args)
                for channel in range(19):
                    logger_child = Logger(args.runs, args)
                    data["channel"] = channel
                    print(f"Running on Channel: {elec_names[channel]}")
                    logger_child.reset(args.runs, args)
                    for run in range(args.runs):
                        import gc
                        gc.collect()
                        seed_everything(run)
                        if args.model == "svm":
                            """
                            SVM
                            """
                            from trainer_ml import SVMTrainer
                            st = SVMTrainer(args, data)
                            logger_child.add_result(run, st.get_result())
                            logger_main.add_result(run, st.get_result())
                        elif args.model == "knn":
                            """
                            KNN
                            """
                            from trainer_ml import KNNTrainer
                            kt = KNNTrainer(args, data)
                            logger_child.add_result(run, kt.get_result())
                            logger_main.add_result(run, kt.get_result())
                        elif args.model == "rf":
                            """
                            Random Forest
                            """
                            from trainer_ml import RFTrainer
                            rt = RFTrainer(args, data)
                            logger_child.add_result(run, rt.get_result())
                            logger_main.add_result(run, rt.get_result())
                        elif args.model == "xgb":
                            """
                            XGBoosting
                            """
                            from trainer_ml import XGBTrainer_riemann
                            xgbt = XGBTrainer(args, data)
                            logger_child.add_result(run, xgbt.get_result())
                            logger_main.add_result(run, xgbt.get_result())
                    logger_child.print_statistics()
                print("Total Value:")
                logger_main.print_statistics()
            else:
                logger = Logger(args.runs, args)
                for run in range(args.runs):
                    import gc
                    gc.collect()
                    seed_everything(run)
                    if args.model == "bsit":
                        """
                        Brain Subject Identification Trainer
                        """
                        from trainer import BSITrainer
                        bt = BSITrainer(args, data)
                        logger.add_result(run, bt.get_result())
                    elif args.model == "eegnet_old":
                        """
                        EEGNet_old
                        """
                        from trainer import EEGNet_old_Trainer
                        eot = EEGNet_old_Trainer(args, data)
                        logger.add_result(run, eot.get_result())
                    elif args.model == "eegnet":
                        """
                        EEGNet
                        """
                        from trainer import EEGNetTrainer
                        et = EEGNetTrainer(args, data)
                        logger.add_result(run, et.get_result())
                    elif args.model == "shallow_net":
                        """
                        ShallowNet
                        """
                        from trainer import ShallowNetTrainer
                        st = ShallowNetTrainer(args, data)
                        logger.add_result(run, st.get_result())
                    elif args.model == "deep_net":
                        """
                        DeepNet
                        """
                        from trainer import DeepNetTrainer
                        dt = DeepNetTrainer(args, data)
                        logger.add_result(run, dt.get_result())
                    elif args.model == "knn":
                        """
                        KNN riemann
                        """
                        from trainer import KNNTrainer_riemann
                        kt = KNNTrainer_riemann(args, data)
                        logger.add_result(run, kt.get_result())
                        if args.plot_confusion_matrix:
                            plot_cm(kt.get_confusion_matrix(), args, data)
                    elif args.model == "rf":
                        """
                        Random Forest riemann
                        """
                        from trainer import RFTrainer_riemann
                        rt = RFTrainer_riemann(args, data)
                        logger.add_result(run, rt.get_result())
                        if args.plot_confusion_matrix:
                            plot_cm(rt.get_confusion_matrix(), args, data)
                    elif args.model == "gb":
                        """
                         Gradient Boosting riemann
                        """
                        from trainer import GBTrainer_riemann
                        gt = GBTrainer_riemann(args, data)
                        logger.add_result(run, gt.get_result())
                        if args.plot_confusion_matrix:
                            plot_cm(gt.get_confusion_matrix(), args, data)
                    elif args.model == "svm":
                        """
                         SVM riemann
                        """
                        from trainer import SVMTrainer_riemann
                        st = SVMTrainer_riemann(args, data)
                        logger.add_result(run, st.get_result())
                        if args.plot_confusion_matrix:
                            plot_cm(st.get_confusion_matrix(), args, data)
                logger.print_statistics()

    """
    Visualizing results
    """
    if args.vis:
        from visualize import Visualizer
        logger = Logger_Channel(args.runs, args)
        for run in range(args.runs):
            import gc
            gc.collect()
            seed_everything(run)
            vt = Visualizer(args, data)
            logger.add_result(run, vt.get_result())
        logger.print_statistics()
        if args.vis_type == 'analysis_activations' or 'deep_dream':
            activations, std = logger.return_data()
            activations_normalized = (activations - numpy.min(activations))/(numpy.max(activations) - numpy.min(activations))
            for i in range(19):
              std[i] = activations_normalized[i]*std[i]/activations[i]
            plot_data(activations_normalized, std, data, args)

    """
    Intruder Detection
    """
    if args.intruder:
        seed_everything(args.seed)
        if args.model == "bsit":
            """
            Brain Subject Identification Trainer
            """
            from trainer import BSITrainer
            bt = BSITrainer(args, data)
        elif args.model == "eegnet_old":
            """
            EEGNet_old
            """
            from trainer import EEGNet_old_Trainer
            eot = EEGNet_old_Trainer(args, data)
        elif args.model == "eegnet":
            """
            EEGNet
            """
            from trainer import EEGNetTrainer
            et = EEGNetTrainer(args, data)
        elif args.model == "shallow_net":
            """
            ShallowNet
            """
            from trainer import ShallowNetTrainer
            st = ShallowNetTrainer(args, data)
        elif args.model == "deep_net":
            """
            DeepNet
            """
            from trainer import DeepNetTrainer
            dt = DeepNetTrainer(args, data)
        elif args.model == "knn":
            """
            KNN riemann
            """
            from trainer import KNNTrainer_riemann
            kt = KNNTrainer_riemann(args, data)
        elif args.model == "rf":
            """
            Random Forest riemann
            """
            from trainer import RFTrainer_riemann
            rt = RFTrainer_riemann(args, data)
        elif args.model == "gb":
            """
             Gradient Boosting riemann
            """
            from trainer import GBTrainer_riemann
            gt = GBTrainer_riemann(args, data)
        elif args.model == "svm":
            """
             SVM riemann
            """
            from trainer import SVMTrainer_riemann
            st = SVMTrainer_riemann(args, data)

def plot_cm(X, args, data):
    X = X.astype('float') / X.sum(axis=1)
    df_cm = pd.DataFrame(X, index = [i for i in range(X.shape[0])],
                  columns = [i for i in range(X.shape[0])])
    fig = plt.figure(figsize = (15,15))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(f'../plots/cm_{args.model}_{args.frequency_bands}_{data["filename"]}.png')
    plt.close()

def plot_data(mean, std, data, args):
    fig = plt.figure(figsize =(4,4))
    ax = fig.add_subplot(1,1,1)
    ax,_=plot_topomap(mean,
                      SENSORS_POS,
                      show=True,
                      cmap="viridis",
                      names=CHANNEL_NAMES,
                      show_names=True,
                      contours=0,
                      extrapolate='head',
                      axes=ax,
                      vmax = numpy.max(mean),
                      vmin = numpy.min(mean),
                      sphere=(0, 0, 0, 1.1))

    cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)
    fig.savefig(f'../plots/{args.vis_type}_feature_map_mean_{data["filename"]}.png')
    plt.close()


    fig = plt.figure(figsize =(4,4))
    ax = fig.add_subplot(1,1,1)
    ax,_=plot_topomap(std,
                      SENSORS_POS,
                      show=True,
                      cmap="Reds",
                      names=CHANNEL_NAMES,
                      show_names=True,
                      contours=0,
                      extrapolate='head',
                      axes=ax,
                      vmax = numpy.max(std),
                      vmin = numpy.min(std),
                      sphere=(0, 0, 0, 1.1))

    cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)
    fig.savefig(f'../plots/{args.vis_type}_feature_map_std_{data["filename"]}.png')
    plt.close()

if __name__ == '__main__':
    main()
