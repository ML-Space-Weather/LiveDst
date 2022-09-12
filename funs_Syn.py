# system
from multiprocessing import cpu_count, Pool
import time
import datetime as dt
import os
import subprocess
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

# data format
import pandas as pd
import numpy as np
import h5py

from scipy.special import erfinv, erf
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import shift
from scipy.stats import pearsonr

# visualize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import pylab 

# ML
import torch
from torch.optim import Adam, AdamW, RMSprop
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import ProgressBar, Checkpoint, EpochScoring
from skorch.dataset import CVSplit, ValidSplit
from skorch.callbacks import EarlyStopping, WarmRestartLR, LRScheduler

# self-made ML
from nets import CNN_1D, PhysinformedNet_AR_2D, lstm_reg, lstm_gp, my_weight_rmse_CB
from nets import cdf_AH, norm_cdf, PhysinformedNet_AR_2D
from nets import seed_torch, init_weights, PhysinformedNet, my_custom_loss_func
from nets import MLP, init_weights, PhysinformedNet_AR
from nets import my_callbacks, seed_torch, maxmin_scale, std_scale

# debug
from tqdm import tqdm
from progressist import ProgressBar as Bar_ori
from ipdb import set_trace as st


def est_beta(X, y, y_real):

    d = y_real - y
    d = (d - d.mean())/d.std()
    N = X.shape[0]
    i = np.arange(1, N+1)
    # RS_min_1 = 1/(np.sqrt(np.pi)*N)
    RS_min_2 = -1*erfinv((2*i-1)/N-1)**2
    # RS_min_3 = np.sqrt(2/np.pi)/2

    RS_min = np.nansum(np.exp(RS_min_2)/np.sqrt(np.pi)/N)
    # CRPS from matlab code
    sigma = np.abs(d)/np.sqrt(np.log(2))+1e-6
    dum = d/sigma/np.sqrt(2)
    CRPS_min = np.nanmean(sigma*(np.sqrt(2)*dum*erf(dum)
                            + np.sqrt(2/np.pi)*np.exp(-1*dum**2) 
                            - 1/np.sqrt(np.pi)))

    return RS_min/(RS_min+CRPS_min), CRPS_min, RS_min


############################################ for batch boost ######################################

def train_Dst_boost_batch(X_t, delay, Dst_sel, ratio,
              iter_num, boost_num, idx_storm, device, 
              criteria, win_size):

    callname = 'Res/'+str(boost_num)+'/'+\
        str(ratio)+\
        '/params_new_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        str(iter_num)+\
        criteria+'.pt'

    my_callbacks = [Checkpoint(f_params=callname),
                    LRScheduler(WarmRestartLR),
                    # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                    EarlyStopping(patience=10),
                    ProgressBar()]

    with h5py.File('Data/mean_std_'+str(delay)+
        '_'+str(Dst_sel)+'_'+str(boost_num)+'.h5', 'r') as f:

        mean_Y = np.array(f['Y_mean'+str(iter_num)])
        std_Y = np.array(f['Y_std'+str(iter_num)])
        mean_X = np.array(f['X_mean'+str(iter_num)])
        std_X = np.array(f['X_std'+str(iter_num)])
        f.close()


    seed_torch(2333)

    hidden_size = 32
    output_size = 1
    input_size = X_t.shape[-1]
    
    X_t = (X_t - mean_X)/std_X

    seed_torch(1029)
    net = lstm_reg(input_size,
                   hidden_size,
                   num_layers=2,
                   output_size=output_size)
    
    net.apply(init_weights)
    
    # st()
    net_regr = PhysinformedNet(
        module=net,
        # module=DDP(net),
        # module=DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5]),
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks,
        optimizer__weight_decay=np.exp(-4),
        # thres=Y_thres,
        thres=.5,
        device=device,  # uncomment this to train with CUDA
        # device='cuda',  # uncomment this to train with CUDA
    )

    # X_t = torch.from_numpy(X_t).float()

    net_regr.initialize()
    net_regr.load_params(f_params=callname)    

    y_pred_t = np.zeros([X_t.shape[0], X_t.shape[1]-win_size+1, win_size])    

    # st()
    for i in tqdm(range(win_size, X_t.shape[1])):
        y_pred_t[:, i-win_size] = net_regr.predict(torch.from_numpy(
                                                X_t[:, i-6:i]).float()).squeeze()
        # st()
        X_t[:, i, -1] = y_pred_t[:, i-win_size, -1]
    # st()
    y_pred_t = y_pred_t*std_Y+mean_Y

    return y_pred_t


def train_std_boost(X, X_t, y, y_real, delay, Dst_sel, \
    ratio, iter_num, boost_num, idx_storm, device, 
    pred, 
    criteria, 
    train=True):

    callname = 'Res/'+str(boost_num)+'/'+\
        str(ratio)+\
        '/params_std_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        str(iter_num)+pred+\
        criteria+'.pt'

    # st()
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=10),
                   ProgressBar()]

    with h5py.File('Data/mean_std_'+str(delay)+
        '_'+str(Dst_sel)+'_'+str(boost_num)+'.h5', 'r') as f:

        mean_Y = np.array(f['Y_mean'+str(iter_num)])
        std_Y = np.array(f['Y_std'+str(iter_num)])
        mean_X = np.array(f['X_mean'+str(iter_num)])
        std_X = np.array(f['X_std'+str(iter_num)])
        max_X = np.array(f['X_max'+str(iter_num)])
        min_X = np.array(f['X_min'+str(iter_num)])
        f.close()

    # st()
    beta, CRPS_min, RS_min = 0.3, 0.3, 0.3
    X_t = (X_t-min_X.min(axis=0))/(max_X.max(axis=0)-min_X.min(axis=0))

    # beta, _, _ = est_beta(X, y, y_real)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.plot(y[:100], 'b.-', label='persistence')
    # ax.plot(y_real[:100], 'm.-', label='real')
    # fig.savefig('Figs/test2.jpg')
    # plt.close()
    
    # y = (y-mean_y)/std_y
    # y_real = (y_real-mean_y)/std_y
    ################# design the model ###################

    seed_torch(1029)
    net = MLP(X.shape[1], 0.1)

    net.apply(init_weights)
    # print('CRPS_min: {}, RS_min: {}'.format(CRPS_min, RS_min))

    net_regr = PhysinformedNet_AR(
        module=net,
        # module=DataParallel(net, device_ids=[0, 1, 2]),
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks_AH,
        optimizer__weight_decay=np.exp(-8),
        beta = beta,
        CRPS_min = CRPS_min,
        RS_min = RS_min,
        mean = mean_Y,
        std = std_Y,
        # d = d,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
    )
    X = torch.from_numpy(X).float()
    X_t = torch.from_numpy(X_t).float()
    # st()
    Y = np.vstack([y.T, y_real.T]).T
    Y = torch.from_numpy(Y).float()

    # st()
    # import ipdb;ipdb.set_trace()
    if train:
        # y_pred = clf.predict(X_t)
        net_regr.fit(X, Y)
        # import ipdb;ipdb.set_trace()
        net_regr.load_params(f_params=callname)
    else:
        net_regr.initialize()
        net_regr.load_params(f_params=callname)  

    std_Y = np.exp(net_regr.predict(X_t).squeeze())
    # std_Y = np.exp((net_regr.predict(X_t).squeeze()-mean_y)/std_y)

    return std_Y


def train_std_GRU_boost(X, X_t, y, y_real, y_t, y_real_t,\
    delay, Dst_sel, \
    ratio, iter_boost, boost_num, idx_storm, 
    device, pred, 
    criteria,
    train=True):

    callname = 'Res/'+str(boost_num)+'/'+\
        str(ratio)+\
        '/params_std_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        str(iter_boost)+pred+\
        criteria+'.pt'

    hidden_size = 32
    output_size = 1
    input_size = X.shape[2]
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=10),
                   ProgressBar()]

    with h5py.File('Data/mean_std_'+str(delay)+
        '_'+str(Dst_sel)+'_'+str(boost_num)+'.h5', 'r') as f:

        mean_Y = np.array(f['Y_mean'+str(iter_boost)])
        std_Y = np.array(f['Y_std'+str(iter_boost)])
        mean_X = np.array(f['X_mean'+str(iter_boost)])
        std_X = np.array(f['X_std'+str(iter_boost)])
        f.close()
    
    X = (X - mean_X)/std_X
    X_t = (X_t - mean_X)/std_X

    # X = (X-min_X)/(max_X-min_X)
    # X_t = (X_t-min_X)/(max_X-min_X)
    beta, CRPS_min, RS_min = 0.3, 0.3, 0.3
    # st()

    ################# design the model ###################

    seed_torch(1029)
    net = lstm_reg(input_size,
                   hidden_size,
                   num_layers=2,
                   output_size=output_size)
    
    net.apply(init_weights)
    # print('CRPS_min: {}, RS_min: {}'.format(CRPS_min, RS_min))

    net_regr = PhysinformedNet_AR_2D(
        module=net,
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks_AH,
        optimizer__weight_decay=np.exp(-8),
        beta = beta,
        CRPS_min = CRPS_min,
        RS_min = RS_min,
        mean = mean_Y,
        std = std_Y,
        # d = d,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
    )
    X = torch.from_numpy(X).float()
    X_t = torch.from_numpy(X_t).float()
    
    # st()
    Y = np.stack([y.squeeze(), y_real.squeeze()], axis=-1)
    Y = torch.from_numpy(Y).float()
    st()
    if train:
        # y_pred = clf.predict(X_t)
        net_regr.fit(X, Y)
        # import ipdb;ipdb.set_trace()
        net_regr.load_params(f_params=callname)
    else:
        net_regr.initialize()
        net_regr.load_params(f_params=callname)  

    std_Y = np.exp(net_regr.predict(X_t)[:, -1].squeeze())
    # std_Y = np.exp((net_regr.predict(X_t).squeeze()-mean_y)/std_y)

    return std_Y


############################################ for boost ######################################

def train_Dst_boost(X, Y, X_t, delay, Dst_sel, ratio,
              iter_num, boost_num, idx_storm, device, 
              criteria,
              train=False):

    callname = 'Res/'+str(boost_num)+'/'+\
        str(ratio)+\
        '/params_new_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        str(iter_num)+\
        criteria+'.pt'

    my_callbacks = [Checkpoint(f_params=callname),
                    LRScheduler(WarmRestartLR),
                    # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                    EarlyStopping(patience=10),
                    ProgressBar()]

    with h5py.File('Data/mean_std_'+str(delay)+
        '_'+str(Dst_sel)+'_'+str(boost_num)+'.h5', 'r') as f:

        mean_Y = np.array(f['Y_mean'+str(iter_num)])
        std_Y = np.array(f['Y_std'+str(iter_num)])
        mean_X = np.array(f['X_mean'+str(iter_num)])
        std_X = np.array(f['X_std'+str(iter_num)])
        f.close()


    seed_torch(2333)

    n_epochs = 50000
    n_iters = 10
    hidden_size = 32
    output_size = 1
    input_size = X.shape[-1]
    
    # st()
    Y = (Y - mean_Y)/std_Y
    # Y_t = (Y_t - mean_Y)/Y_std
    
    X = (X - mean_X)/std_X
    X_t = (X_t - mean_X)/std_X

    # import ipdb;ipdb.set_trace()
    seed_torch(1029)
    net = lstm_reg(input_size,
                   hidden_size,
                   num_layers=2,
                   output_size=output_size)
    '''    
    net = CNN_1D(num_channel=input_size, out=output_size)

    '''
    net.apply(init_weights)
    
    # st()
    net_regr = PhysinformedNet(
        module=net,
        # module=DDP(net),
        # module=DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5]),
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks,
        optimizer__weight_decay=np.exp(-4),
        # thres=Y_thres,
        thres=.5,
        device=device,  # uncomment this to train with CUDA
        # device='cuda',  # uncomment this to train with CUDA
    )

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X_t = torch.from_numpy(X_t).float()

    # net_regr.callbacks = my_callbacks2
    if train:
        # y_pred = clf.predict(X_t)
        net_regr.fit(X, Y)
        # st()
        net_regr.load_params(f_params=callname)
    else:
        net_regr.initialize()
        net_regr.load_params(f_params=callname)        

    # st()
    y_pred_t = net_regr.predict(X_t)#.reshape(-1, 1)
    # st()
    y_pred_t = y_pred_t*std_Y+mean_Y

    return y_pred_t


def train_std_boost_batch(X_t, delay, Dst_sel, \
    ratio, iter_num, boost_num, idx_storm, device, 
    pred, criteria, win_size):

    callname = 'Res/'+str(boost_num)+'/'+\
        str(ratio)+\
        '/params_std_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        str(iter_num)+pred+\
        criteria+'.pt'

    # st()
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=10),
                   ProgressBar()]

    with h5py.File('Data/mean_std_'+str(delay)+
        '_'+str(Dst_sel)+'_'+str(boost_num)+'.h5', 'r') as f:

        mean_Y = np.array(f['Y_mean'+str(iter_num)])
        std_Y = np.array(f['Y_std'+str(iter_num)])
        mean_X = np.array(f['X_mean'+str(iter_num)])
        std_X = np.array(f['X_std'+str(iter_num)])
        max_X = np.array(f['X_max'+str(iter_num)])
        min_X = np.array(f['X_min'+str(iter_num)])
        f.close()

    # st()
    beta, CRPS_min, RS_min = 0.3, 0.3, 0.3
    ################# design the model ###################
    X_t = (X_t-min_X.min(axis=0))/(max_X.max(axis=0)-min_X.min(axis=0))

    seed_torch(1029)
    net = MLP(X_t.shape[2], 0.1)

    net.apply(init_weights)
    # print('CRPS_min: {}, RS_min: {}'.format(CRPS_min, RS_min))

    net_regr = PhysinformedNet_AR(
        module=net,
        # module=DataParallel(net, device_ids=[0, 1, 2]),
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks_AH,
        optimizer__weight_decay=np.exp(-8),
        beta = beta,
        CRPS_min = CRPS_min,
        RS_min = RS_min,
        mean = mean_Y,
        std = std_Y,
        # d = d,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
    )
    
    net_regr.initialize()
    net_regr.load_params(f_params=callname)  

    X_t = torch.from_numpy(X_t).float()

    std_Y = np.zeros([X_t.shape[0], X_t.shape[1]-win_size+1])
    
    # st()
    for i in tqdm(range(win_size, X_t.shape[1])):
        std_Y[:, i-win_size] = np.exp(net_regr.predict(X_t[:, i]).squeeze())
    # std_Y = np.exp((net_regr.predict(X_t).squeeze()-mean_y)/std_y)

    return std_Y


def train_std_GRU_boost_batch(X_t,
    delay, Dst_sel, \
    ratio, iter_boost, boost_num, idx_storm, 
    device, pred, 
    criteria,
    win_size):

    callname = 'Res/'+str(boost_num)+'/'+\
        str(ratio)+\
        '/params_std_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        str(iter_boost)+pred+\
        criteria+'.pt'

    hidden_size = 32
    output_size = 1
    input_size = X_t.shape[2]
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=10),
                   ProgressBar()]

    with h5py.File('Data/mean_std_'+str(delay)+
        '_'+str(Dst_sel)+'_'+str(boost_num)+'.h5', 'r') as f:

        mean_Y = np.array(f['Y_mean'+str(iter_boost)])
        std_Y = np.array(f['Y_std'+str(iter_boost)])
        mean_X = np.array(f['X_mean'+str(iter_boost)])
        std_X = np.array(f['X_std'+str(iter_boost)])
        f.close()
    
    X_t = (X_t - mean_X)/std_X

    # X = (X-min_X)/(max_X-min_X)
    # X_t = (X_t-min_X)/(max_X-min_X)
    beta, CRPS_min, RS_min = 0.3, 0.3, 0.3
    # st()

    ################# design the model ###################

    seed_torch(1029)
    net = lstm_reg(input_size,
                   hidden_size,
                   num_layers=2,
                   output_size=output_size)
    
    net.apply(init_weights)
    # print('CRPS_min: {}, RS_min: {}'.format(CRPS_min, RS_min))

    net_regr = PhysinformedNet_AR_2D(
        module=net,
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks_AH,
        optimizer__weight_decay=np.exp(-8),
        beta = beta,
        CRPS_min = CRPS_min,
        RS_min = RS_min,
        mean = mean_Y,
        std = std_Y,
        # d = d,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
    )
    X_t = torch.from_numpy(X_t).float()
    
    net_regr.initialize()
    net_regr.load_params(f_params=callname)  
    std_Y = np.zeros([X_t.shape[0], X_t.shape[1]-win_size+1])
    
    # st()
    for i in tqdm(range(win_size, X_t.shape[1])):
        # st()
        std_Y[:, i-win_size] = np.exp(net_regr.predict(X_t[:, i-win_size:i])[:, -1]).squeeze()
    # std_Y = np.exp(net_regr.predict(X_t)[:, -1].squeeze())
    # std_Y = np.exp((net_regr.predict(X_t).squeeze()-mean_y)/std_y)

    return std_Y



def GP(X, Y, X_t, Y_t, 
       device, figname):
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y_t.shape[1]).to(device)
    # model = MultitaskGPModel(X_t, Y_t, likelihood).to(device)
    model = MultitaskGPModel(X_t, Y_t[test_idx, :, 0].T, likelihood).to(device)
    # print(Y_t[test_idx, :, j].shape)

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=1e-3)

    # Try to change the number of iterations!
    training_iter=200  # We need now a larger number of iterations for training
    # gpytorch.settings.cholesky_jitter.value = 1e-1

    for i in tqdm(range(training_iter)):
        # for j in range(1):
        for j in range(Y_t.shape[2]):
        # for j in range(Y_t.shape[2]):
            optimizer.zero_grad()
            # st()
            output = model(X_t)
            # print(Y_t[test_idx, :].shape)
            # print(output)
            loss = -mll(output, Y_t[test_idx, :, j].T).to(device)
            # loss = -mll(output, Y_t[test_idx]).to(device)
            loss.backward()
            optimizer.step()
        if i % 10 == 0:
            print('after {}th iteration, loss is {}'.format(i, loss))