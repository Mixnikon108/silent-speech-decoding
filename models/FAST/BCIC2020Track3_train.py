"""
Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.
Currently under review for publication.
Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
"""

# ──────────────────────────────────────────────────────────────────────────────
# SIN CAMBIOS EN LOS IMPORTS ORIGINALES
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, argparse, random, time, logging
import numpy as np
import torch
torch.set_num_threads(8)
import torch.optim as optim
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import h5py, einops
from sklearn.model_selection import KFold
from transformers import PretrainedConfig
import lightning as pl
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

from FAST import FAST as Tower
from utils import green, yellow
from BCIC2020Track3_preprocess import Electrodes, Zones

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES (sin cambios, salvo donde se indica)
# ──────────────────────────────────────────────────────────────────────────────
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def load_standardized_h5(cache_fn):
    X, Y = [], []
    with h5py.File(cache_fn, 'r') as f:
        subjects = list(f.keys())
        for subject in subjects:
            X.append(f[subject]['X'][()])
            Y.append(f[subject]['Y'][()])
    X, Y = np.array(X), np.array(Y)
    print('Loaded from', cache_fn, X.shape, Y.shape)
    return X, Y

def inference_on_loader(model, loader):
    model.eval(); model.cuda()
    with torch.no_grad():
        Pred, Real = [], []
        for x, y in loader:
            preds = torch.argmax(model(x.cuda()), dim=1).cpu()
            Pred.append(preds); Real.append(y)
        Pred, Real = torch.cat(Pred), torch.cat(Real)
    return Pred.numpy(), Real.numpy()

class BasicDataset(Dataset):
    def __init__(self, data, label):
        if len(data.shape) == 4:                  # (folds, trials, ch, samples)
            data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        self.data  = torch.from_numpy(data).float()
        self.labels= torch.from_numpy(label).long()
    def __len__(self):          return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class EEG_Encoder_Module(pl.LightningModule):
    def __init__(self, config, max_epochs, niter_per_ep):
        super().__init__()
        self.config = config
        self.model  = Tower(config)
        self.loss   = nn.CrossEntropyLoss()
        self.cosine_lr_list = cosine_scheduler(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=10)
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
    def configure_optimizers(self):
        self.optimizer  = optim.AdamW(self.parameters(), lr=0.0005)
        self.scheduler  = optim.lr_scheduler.LambdaLR(self.optimizer,
                              lambda epoch: self.cosine_lr_list[self.global_step-1])
        return [self.optimizer],[{'scheduler': self.scheduler,'interval':'step'}]
    def training_step(self, batch, batch_idx):
        x,y = batch
        pred= self.model(x)
        return self.loss(pred,y)

# ──────────────────────────────────────────────────────────────────────────────
# NUEVA VERSIÓN DE Finetune
#   • mantiene csv-Tune (como antes)
#   • añade csv-Test (inferencia sobre 50 trials de test)
# ──────────────────────────────────────────────────────────────────────────────
def Finetune(config, Data_X, Data_Y,     # train+val del sujeto
             Data_X_test, Data_Y_test,   # NEW: 50 trials de test
             logf_tune, logf_test,       # rutas csv
             max_epochs=200, ckpt_pretrain=None):

    seed_all(42)
    # ============ 5-fold interna sobre train+val (csv-Tune) ============ #
    Pred, Real = [], []
    kf = KFold(n_splits=5, shuffle=False)
    for _train_idx, _test_idx in kf.split(Data_X):
        x_train, y_train = Data_X[_train_idx], Data_Y[_train_idx]
        x_test , y_test  = Data_X[_test_idx], Data_Y[_test_idx]

        train_loader = DataLoader(BasicDataset(x_train, y_train),
                                  batch_size=len(x_train), shuffle=True,  num_workers=0, pin_memory=True)
        test_loader  = DataLoader(BasicDataset(x_test , y_test ),
                                  batch_size=len(x_test ), shuffle=False, num_workers=0, pin_memory=True)

        model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
        if ckpt_pretrain is not None:
            model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

        print(yellow(logf_tune), green(ckpt_pretrain),
              x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        trainer = pl.Trainer(strategy='auto', accelerator='gpu', devices=[args.gpu],
                             max_epochs=max_epochs, enable_progress_bar=False,
                             enable_checkpointing=False, precision='bf16-mixed', logger=False)
        trainer.fit(model, train_dataloaders=train_loader)

        pred, real = inference_on_loader(model.model, test_loader)
        Pred.append(pred); Real.append(real)

    Pred, Real = np.concatenate(Pred), np.concatenate(Real)
    np.savetxt(logf_tune, np.column_stack([Pred, Real]), delimiter=',', fmt='%d')

    # ============ ENTRENAR EN TODO train+val y PROBAR EN TEST ============ #
    train_loader_full = DataLoader(BasicDataset(Data_X, Data_Y),
                                   batch_size=len(Data_X), shuffle=True, num_workers=0, pin_memory=True)
    test_loader_ext  = DataLoader(BasicDataset(Data_X_test, Data_Y_test),
                                  batch_size=len(Data_X_test), shuffle=False, num_workers=0, pin_memory=True)

    model_ext = EEG_Encoder_Module(config, max_epochs, len(train_loader_full))
    if ckpt_pretrain is not None:
        model_ext.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

    print(yellow(logf_test), green("external-test"),
          Data_X.shape, Data_Y.shape, Data_X_test.shape, Data_Y_test.shape)

    trainer_ext = pl.Trainer(strategy='auto', accelerator='gpu', devices=[args.gpu],
                             max_epochs=max_epochs, enable_progress_bar=False,
                             enable_checkpointing=False, precision='bf16-mixed', logger=False)
    trainer_ext.fit(model_ext, train_dataloaders=train_loader_full)

    pred_ext, real_ext = inference_on_loader(model_ext.model, test_loader_ext)
    np.savetxt(logf_test, np.column_stack([pred_ext, real_ext]), delimiter=',', fmt='%d')

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gpu',   type=int,  default=0)
    args.add_argument('--folds', type=str,  default='0-15')
    args = args.parse_args()

    if '-' in args.folds:
        start,end  = [int(x) for x in args.folds.split('-')]
        args.folds = list(range(start, end))
    else:
        args.folds = [int(x) for x in args.folds.split(',')]

    Run = "Results/FAST/"
    os.makedirs(Run, exist_ok=True)

    # ─────────── Config del modelo (igual) ───────────
    sfreq = 256  # Modificado a 256 Hz para BCI2020 <<<<<<< original 250 Hz
    config = PretrainedConfig(
        electrodes   = Electrodes,
        zone_dict    = Zones,
        dim_cnn      = 32,
        dim_token    = 32,
        seq_len      = 800,
        window_len   = sfreq,
        slide_step   = sfreq//2,
        head         = 'Conv4Layers',
        n_classes    = 5,
        num_layers   = 4,
        num_heads    = 8,
        dropout      = 0.1,
    )

    # ─────────── Carga train+val (h5) ───────────
    X, Y = load_standardized_h5('/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/Processed/BCI2020.h5')

    # ─────────── Carga TEST externo (npz) ───────────
    npz = np.load('/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/filtered_BCI2020.npz')
    X_test_all = npz['X_test']      # (795, 64, 750)
    y_test_all = npz['y_test']      # (5 , 750)

    # helper para convertir test-npz → shape (50,64,800)
    def get_subject_test(k):
        xs = X_test_all[:, :, k*50:(k+1)*50]              # (795,64,50)
        xs = xs.transpose(2,1,0)                          # (50,64,795)
        if xs.shape[2] < 800:                             # pad a 800
            xs = np.pad(xs, ((0,0),(0,0),(0,800-xs.shape[2])), 'constant')
        ys = y_test_all[:, k*50:(k+1)*50]                 # (5,50)
        ys = np.argmax(ys, axis=0).astype(np.int64)       # (50,)
        return xs, ys

    # ─────────── Loop por sujetos (folds) ───────────
    for fold in range(15):
        if fold not in args.folds: continue
        flog_tune = f"{Run}/{fold}-Tune.csv"
        flog_test = f"{Run}/{fold}-Test.csv"

        if os.path.exists(flog_tune) and os.path.exists(flog_test):
            print(f"Skip {fold} (csv existentes)")
            continue

        x_test_subj, y_test_subj = get_subject_test(fold)
        Finetune(config,
                 X[fold], Y[fold],                 # train+val
                 x_test_subj, y_test_subj,         # NEW external test
                 flog_tune, flog_test,             # logs
                 max_epochs=200)

    # ─────────── Calcular accuracies ───────────
    acc_tune, acc_test = [], []
    for fold in range(15):
        flog_tune = f"{Run}/{fold}-Tune.csv"
        flog_test = f"{Run}/{fold}-Test.csv"
        if not (os.path.exists(flog_tune) and os.path.exists(flog_test)):
            print(f"Skip acc fold {fold}")
            continue
        pred_t, lbl_t = np.loadtxt(flog_tune, delimiter=',', dtype=int).T
        pred_e, lbl_e = np.loadtxt(flog_test, delimiter=',', dtype=int).T
        acc_tune.append(np.mean(pred_t == lbl_t))
        acc_test.append(np.mean(pred_e == lbl_e))

    print(f"Tune  Accuracy: {np.mean(acc_tune):.3f} ± {np.std(acc_tune):.3f}")
    print(f"Test  Accuracy: {np.mean(acc_test):.3f} ± {np.std(acc_test):.3f}")








# """
# Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
# Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.
# Currently under review for publication.
# Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
# """

# import os
# import sys
# import argparse
# import random
# import time
# import numpy as np
# import torch
# torch.set_num_threads(8)
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torchmetrics
# import logging
# import h5py
# import einops
# from sklearn.model_selection import KFold
# from transformers import PretrainedConfig
# import lightning as pl
# logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
# logging.getLogger('lightning').setLevel(logging.WARNING)

# from FAST import FAST as Tower
# from utils import green, yellow
# from BCIC2020Track3_preprocess import Electrodes, Zones

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.set_float32_matmul_precision('medium')

# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule

# def load_standardized_h5(cache_fn):
#     X, Y = [], []
#     with h5py.File(cache_fn, 'r') as f:
#         subjects = list(f.keys())
#         for subject in subjects:
#             X.append(f[subject]['X'][()])
#             Y.append(f[subject]['Y'][()])
#     X, Y = np.array(X), np.array(Y)
#     print('Loaded from', cache_fn, X.shape, Y.shape)
#     return X, Y

# def inference_on_loader(model, loader):
#     model.eval()
#     model.cuda()
#     with torch.no_grad():
#         Pred, Real = [], []
#         for x, y in loader:
#             preds = torch.argmax(model(x.cuda()), dim=1).cpu()
#             Pred.append(preds)
#             Real.append(y)
#         Pred, Real = torch.cat(Pred), torch.cat(Real)
#     return Pred.numpy(), Real.numpy()

# class BasicDataset(Dataset):
#     def __init__(self, data, label):
#         if len(data.shape) == 4:
#             data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
#         self.data, self.labels = torch.from_numpy(data), torch.from_numpy(label)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample, label = self.data[idx], self.labels[idx]
#         return sample, label

# class EEG_Encoder_Module(pl.LightningModule):
#     def __init__(self, config, max_epochs, niter_per_ep):
#         super().__init__()
#         self.config = config
#         self.model = Tower(config)
#         self.loss = nn.CrossEntropyLoss()
#         self.cosine_lr_list = cosine_scheduler(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=10)
#         self.accuracy = torchmetrics.Accuracy('multiclass', num_classes = config.n_classes)

#     def configure_optimizers(self):
#         self.optimizer = optim.AdamW(self.parameters(), lr=0.0005)
#         self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: self.cosine_lr_list[self.global_step-1])
#         return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         pred = self.model(x)
#         return self.loss(pred, y)

# def Finetune(config, Data_X, Data_Y, logf, max_epochs=200, ckpt_pretrain=None):
#     seed_all(42)
#     Pred, Real = [], []
#     kf = KFold(n_splits=5, shuffle=False)
#     for _train_idx, _test_idx in kf.split(Data_X):
#         x_train, y_train = Data_X[_train_idx], Data_Y[_train_idx]
#         x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

#         train_data = BasicDataset(x_train, y_train)
#         train_loader = DataLoader(train_data, batch_size=len(x_train), shuffle=True, num_workers=0, pin_memory=True)
#         test_data = BasicDataset(x_test, y_test)
#         test_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=False, num_workers=0, pin_memory=True)

#         model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
#         if ckpt_pretrain is not None:
#             model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

#         print(yellow(logf), green(ckpt_pretrain), x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#         trainer = pl.Trainer(strategy='auto', accelerator='gpu', devices=[args.gpu], max_epochs=max_epochs, callbacks=[], 
#                             enable_progress_bar=False, enable_checkpointing=False, precision='bf16-mixed', logger=False)
#         trainer.fit(model, train_dataloaders=train_loader)

#         # Test data is used only once
#         pred, real = inference_on_loader(model.model, test_loader)
#         Pred.append(pred)
#         Real.append(real)
#     Pred, Real = np.concatenate(Pred), np.concatenate(Real)
#     np.savetxt(logf, np.array([Pred, Real]).T, delimiter=',', fmt='%d')

# if __name__ == '__main__':
#     args = argparse.ArgumentParser()
#     args.add_argument('--gpu', type=int, default=0)
#     args.add_argument('--folds', type=str, default='0-15')
#     args = args.parse_args()

#     if '-' in args.folds:
#         start, end = [int(x) for x in args.folds.split('-')]
#         args.folds = list(range(start, end))
#     else:
#         args.folds = [int(x) for x in args.folds.split(',')]

#     Run = "Results/FAST/"
#     os.makedirs(f"{Run}", exist_ok=True)

#     sfreq = 250
#     config = PretrainedConfig(
#         electrodes=Electrodes,
#         zone_dict=Zones,
#         dim_cnn=32,
#         dim_token=32,
#         seq_len=800,
#         window_len=sfreq,
#         slide_step=sfreq//2,
#         head='Conv4Layers',
#         n_classes=5,
#         num_layers=4,
#         num_heads=8,
#         dropout=0.1,
#     )
    
#     X, Y = load_standardized_h5('/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/Processed/BCI2020.h5')
#     for fold in range(15):
#         if fold not in args.folds:
#             continue
#         flog = f"{Run}/{fold}-Tune.csv"
#         if os.path.exists(flog):
#             print(f"Skip {flog}")
#             continue
#         Finetune(config, X[fold], Y[fold], flog, max_epochs=200)

#     accuracy = []
#     for fold in range(15):
#         flog = f"{Run}/{fold}-Tune.csv"
#         if not os.path.exists(flog):
#             print(f"Skip {flog}")
#             continue
#         data = np.loadtxt(flog, delimiter=',', dtype=int)
#         pred, label = data[:, 0], data[:, 1]
#         accuracy.append(np.mean(pred == label))

#     print(f"Accuracy: {np.mean(accuracy):3f}, Std: {np.std(accuracy):3f}")