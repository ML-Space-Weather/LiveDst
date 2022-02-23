# system
from multiprocessing import cpu_count, Pool
import time
import datetime as dt

# data format
import pandas as pd
import numpy as np
import h5py

from scipy.special import erfinv, erf
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import shift

# visualize
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab 

# ML
import torch
from torch.optim import Adam, AdamW, RMSprop
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

class MyBar(Bar_ori):
    template = ('Download |{animation}| {done:B}/{total:B}')
    done_char = '⬛'

def smooth(X, width, order=1):
    # window size 51, polynomial order 3
    xhat = savgol_filter(X, width, order)
    return xhat

def stretch(X, ratio = 1.1, thres=-100):
    idx = np.where(X<thres)
    maxi = X.max()
    X[idx] = ratio*(X[idx] - maxi) + maxi
    return X

def est_beta(X, y, y_real):

    d = y_real - y
    # d = (d - d.mean())/d.std()
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


def storm_sel_omni(Omni_data, delay, Dst_sel, width):

    varis = ['N',    
             'V',    
             'BX_GSE',
             'BY_GSE',
             'BZ_GSE',
             'F10_INDEX',
             'DST'
             ]

    sta = ['HON', 'SJG', 'KAK', 'HER']
    
    # Omni read
    df = pd.read_pickle(Omni_data)
    Omni_data = df[varis]
    Omni_date = df.index
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')

    # Fill missing values
    Omni_data.interpolate(inplace=True)
    Omni_data.interpolate(inplace=True)
    Omni_data.dropna(inplace=True)
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')
    N = np.array(Omni_data['N'])
    V = np.array(Omni_data['V'])
    Bx = np.array(Omni_data['BX_GSE'])
    By = np.array(Omni_data['BY_GSE'])
    Bz = np.array(Omni_data['BZ_GSE'])
    F107 = np.array(Omni_data['F10_INDEX'])
    F107 = shift(F107, 24, cval=0)
    Dst = smooth(np.array(Omni_data['DST']), width)
    # Dst = stretch(Dst, ratio=ratio, width=Dst_sel)
    B_norm = np.sqrt(Bx**2+By**2+Bz**2)
    
    ################# SH variables ###################

    DOY = np.asarray([date.timetuple().tm_yday for date in Omni_date])
    year = np.asarray([date.year for date in Omni_date])
    month = np.asarray([date.month for date in Omni_date])
    dom = np.asarray([date.day for date in Omni_date])
    UTC = np.asarray([date.hour for date in Omni_date])
    date_clu = np.vstack([year, month, dom, UTC]).T
    t_year = 23.4*np.cos((DOY-172)*2*np.pi/365.25)
    t_day = 11.2*np.cos((UTC-16.72)*2*np.pi/24)
    t = t_year+t_day
    theta_c = np.arctan2(By, Bz)
    SH_vari = np.vstack([
                         np.sqrt(F107),
                         t,
                         np.sin(theta_c),
    ]).T

    ################## persistence ####################

    Y_Per = shift(Dst, delay, cval=np.NaN)
    error = np.sqrt(np.nanmean((Y_Per - Dst)**2))
    # print('RMSE of all persistence model:{}'.format(error))

    ################## NN ##############################
    X_NN = np.zeros([Dst.shape[0]-6-delay, 13])
    Y_NN = np.zeros(Dst.shape[0]-6-delay)
    date_NN = np.zeros([Dst.shape[0]-6-delay, 4])

    for i in tqdm(np.arange(5, Dst.shape[0]-delay-1)):

        X_NN[i-6] = np.hstack((N[i], V[i], B_norm[i], Bz[i],
                               SH_vari[i],
                               Dst[i-5:i+1]))
        Y_NN[i-6] = Dst[i+delay]
        date_NN[i-6] = date_clu[i+delay+1]
        

    ################## lstm/1D_CNN ##############################
    X_DL = np.zeros([Dst.shape[0]-6-delay, 6, 8])
    Y_DL = np.zeros([Dst.shape[0]-6-delay, 6, 1])
    Y_Per = np.zeros([Dst.shape[0]-6-delay])
    date_DL = np.zeros([Dst.shape[0]-6-delay, 4])

    Dst_win = np.zeros([Dst.shape[0], 6])
    for i in np.arange(6, Dst.shape[0]-delay):

        Dst_win[i-6] = Dst[i-6:i]

    X0 = np.vstack([N, V, B_norm, Bz,
                    SH_vari.T,
                    Dst]).T
    for i in tqdm(np.arange(5, Dst.shape[0]-delay-1)):

        X_DL[i-6] = X0[i-5:i+1]
        Y_DL[i-6] = np.expand_dims(Dst[i-5+delay:i+delay+1],
                                    axis=1)
        Y_Per[i-6] = Dst[i]
        date_DL[i-6] = date_clu[i+delay+1]
    
    peaks, _ = find_peaks(Y_NN*-1,
                          distance=240,
                          width=5)
    peaks = np.hstack((peaks, Y_NN.shape[0]-10))

    idx = np.where(Y_NN[peaks] <= Dst_sel)[0]
    idx_clu = np.zeros([len(idx), 2])

    with h5py.File('Data/data_'+str(delay)+
                   '_'+str(Dst_sel)+'.h5', 'w') as f:
        for i, idx_t in enumerate(idx):

            # print('peak {}:'.format(i), Omni_date[peaks[idx_t]])
            idx_clu[i, 0] = int(np.where(Y_NN[:peaks[idx_t]] >= 0)[0][-1]-24)

            try:
                idx_clu[i, 1] = int(np.where(Y_NN[peaks[idx_t]:] >= 0)[0][0]+24+peaks[idx_t])
            except:
                idx_clu[i, 1] = Y_NN.shape[0]
                
            print('{} & {} & {} & {}'.format(i, 
                                             Omni_date[int(idx_clu[i, 0])],
                                             Omni_date[int(idx_clu[i, 1])],
                                             Y_NN[peaks[idx_t]]))

            f.create_dataset('X_NN_'+str(i),\
                data=X_NN[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('Y_NN_'+str(i),\
                data=np.expand_dims(Y_NN[int(idx_clu[i, 0]):int(idx_clu[i, 1])], axis=1))
            f.create_dataset('date_NN_'+str(i),\
                data=date_NN[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('X_DL_'+str(i),\
                data=X_DL[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('Y_DL_'+str(i),\
                data=Y_DL[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('Dst_Per'+str(i),\
                data=Y_Per[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('date_DL_'+str(i),\
                data=date_DL[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
        f.create_dataset('num', data=i)
    f.close()


def storm_sel_ACE(Omni_data, delay, Dst_sel, width, num):
    
    # Omni read
    df = pd.read_pickle(Omni_data)
    Omni_data = df[24:-24]
    Omni_date = df.index[24:-24]
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')

    # Fill missing values
    df = df.interpolate(method='linear').ffill().bfill()
    # Omni_data.dropna(inplace=True)
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')
    
    Bx = Omni_data['BGSEc._1 (nT)']
    By = Omni_data['BGSEc._2 (nT)']
    Bz = Omni_data['BGSEc._3 (nT)']
    N = Omni_data['Np (#/cc)']
    V = Omni_data['Vp (km/s)']
    F107 = Omni_data['observed_flux (solar flux unit (SFU))']
    F107 = shift(F107, 24, cval=0)
    Dst = smooth(np.array(Omni_data['dst (nT)']), width)
    # Dst = stretch(Dst, ratio=ratio, width=Dst_sel)
    B_norm = np.sqrt(Bx**2+By**2+Bz**2)
    
    ################# SH variables ###################

    DOY = np.asarray([date.timetuple().tm_yday for date in Omni_date])
    year = np.asarray([date.year for date in Omni_date])
    month = np.asarray([date.month for date in Omni_date])
    dom = np.asarray([date.day for date in Omni_date])
    UTC = np.asarray([date.hour for date in Omni_date])
    date_clu = np.vstack([year, month, dom, UTC]).T
    t_year = 23.4*np.cos((DOY-172)*2*np.pi/365.25)
    t_day = 11.2*np.cos((UTC-16.72)*2*np.pi/24)
    t = t_year+t_day
    theta_c = np.arctan2(By, Bz)
    print('shape of t: {}'.format(t.shape))
    print('shape of F107: {}'.format(F107.shape))
    print('shape of theta_c: {}'.format(theta_c.shape))
    SH_vari = np.vstack([
                         np.sqrt(F107),
                         t,
                         np.sin(theta_c),
    ]).T

    ################## persistence ####################

    Y_Per = shift(Dst, delay, cval=np.NaN)
    error = np.sqrt(np.nanmean((Y_Per - Dst)**2))
    # print('RMSE of all persistence model:{}'.format(error))

    ################## NN ##############################
    X_NN = np.zeros([Dst.shape[0]-6-delay, 13])
    Y_NN = np.zeros(Dst.shape[0]-6-delay)
    date_NN = np.zeros([Dst.shape[0]-6-delay, 4])

    for i in tqdm(np.arange(5, Dst.shape[0]-delay-1)):

        X_NN[i-6] = np.hstack((N[i], V[i], B_norm[i], Bz[i],
                               SH_vari[i],
                               Dst[i-5:i+1]))
        Y_NN[i-6] = Dst[i+delay]
        date_NN[i-6] = date_clu[i+delay+1]
        

    ################## lstm/1D_CNN ##############################
    X_DL = np.zeros([Dst.shape[0]-6-delay, 6, 8])
    Y_DL = np.zeros([Dst.shape[0]-6-delay, 6, 1])
    Y_Per = np.zeros([Dst.shape[0]-6-delay])
    date_DL = np.zeros([Dst.shape[0]-6-delay, 4])

    Dst_win = np.zeros([Dst.shape[0], 6])
    for i in np.arange(6, Dst.shape[0]-delay):

        Dst_win[i-6] = Dst[i-6:i]

    X0 = np.vstack([N, V, B_norm, Bz,
                    SH_vari.T,
                    Dst]).T
    for i in tqdm(np.arange(5, Dst.shape[0]-delay-1)):

        X_DL[i-6] = X0[i-5:i+1]
        Y_DL[i-6] = np.expand_dims(Dst[i-5+delay:i+delay+1],
                                    axis=1)
        Y_Per[i-6] = Dst[i]
        date_DL[i-6] = date_clu[i+delay+1]
    
    peaks, _ = find_peaks(Y_NN*-1,
                          distance=240,
                          width=5)
    peaks = np.hstack((peaks, Y_NN.shape[0]-10))
    # st()
    idx = np.where(Y_NN[peaks] <= Dst_sel+10)[0]
    idx_clu = np.zeros([len(idx), 2])

    # print(idx)

    with h5py.File('Data/data_'+str(delay)+
                   '_'+str(Dst_sel)+'.h5', 'a') as f:
        
        for v in ['X_NN_'+str(num),
                  'Y_NN_'+str(num),
                  'date_NN_'+str(num),
                  'X_DL_'+str(num),
                  'Y_DL_'+str(num),
                  'Dst_Per'+str(num),
                  'date_DL_'+str(num)]:
            if v in f:
                del f[v]

        for i, idx_t in enumerate(idx):
            
            # print('peak {}:'.format(i), Omni_date[peaks[idx_t]])
            idx_clu[i, 0] = int(np.where(Y_NN[:peaks[idx_t]] >= 0)[0][-1]-24)

            try:
                idx_clu[i, 1] = int(np.where(Y_NN[peaks[idx_t]:] >= 0)[0][0]+24+peaks[idx_t])
            except:
                idx_clu[i, 1] = Y_NN.shape[0]
                
            print('{} & {} & {} & {}'.format(num, 
                                             Omni_date[int(idx_clu[i, 0])],
                                             Omni_date[int(idx_clu[i, 1])],
                                             Y_NN[peaks[idx_t]]))

            f.create_dataset('X_NN_'+str(num),\
                data=X_NN[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('Y_NN_'+str(num),\
                data=np.expand_dims(Y_NN[int(idx_clu[i, 0]):int(idx_clu[i, 1])], axis=1))
            f.create_dataset('date_NN_'+str(num),\
                data=date_NN[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('X_DL_'+str(num),\
                data=X_DL[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('Y_DL_'+str(num),\
                data=Y_DL[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('Dst_Per'+str(num),\
                data=Y_Per[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
            f.create_dataset('date_DL_'+str(num),\
                data=date_DL[int(idx_clu[i, 0]):int(idx_clu[i, 1])])
    f.close()
    
def train_Dst(X, Y, X_t, delay, Dst_sel, train=True):

    callname = 'Res/params_new_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'.pt'

    my_callbacks = [Checkpoint(f_params=callname),
                    LRScheduler(WarmRestartLR),
                    # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                    EarlyStopping(patience=5),
                    ProgressBar()]

    seed_torch(2333)

    n_epochs = 50000
    n_iters = 10
    hidden_size = 32
    output_size = 1
    input_size = X.shape[2]

    mean_Y = np.mean(Y)
    Y_std = np.std(Y)

    # mean_Y = Y.min()
    # std_Y = Y.max() - Y.min()
    
    Y = (Y - mean_Y)/Y_std
    # Y_t = (Y_t - mean_Y)/Y_std

    mean_X = np.mean(X.reshape(-1, X.shape[2]), axis=0)
    std_X = np.std(X.reshape(-1, X.shape[2]), axis=0)
    
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
    
    net_regr = PhysinformedNet(
        module=net,
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(0.2),
        batch_size=128,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks,
        optimizer__weight_decay=np.exp(-4),
        # thres=Y_thres,
        thres=.5,
        device='cuda',  # uncomment this to train with CUDA
    )

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X_t = torch.from_numpy(X_t).float()

    # net_regr.callbacks = my_callbacks2
    if train:
        # y_pred = clf.predict(X_t)
        net_regr.fit(X, Y)
        # import ipdb;ipdb.set_trace()
        net_regr.load_params(f_params=callname)
    else:
        net_regr.initialize()
        net_regr.load_params(f_params=callname)        

    y_pred_t = net_regr.predict(X_t)#.reshape(-1, 1)
    y_pred_t = y_pred_t*Y_std+mean_Y

    return y_pred_t


def train_std(X, X_t, y, y_real, delay, Dst_sel, \
    idx_storm, device, pred='gru', train=True):

    callname = 'Res/params_std_new2_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        pred+'.pt'
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=5),
                   ProgressBar()]

    max_X = X.max(axis = 0)
    min_X = X.min(axis = 0)

    mean_y = y_real.mean()
    std_y = y_real.std()

    X = (X-min_X)/(max_X-min_X)
    X_t = (X_t-min_X)/(max_X-min_X)
    # st()
    beta, CRPS_min, RS_min = est_beta(X, y, y_real)
    # beta, _, _ = est_beta(X, y, y_real)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(y[:100], 'b.-', label='persistence')
    ax.plot(y_real[:100], 'm.-', label='real')
    fig.savefig('Figs/test2.jpg')
    plt.close()
    
    # y = (y-mean_y)/std_y
    # y_real = (y_real-mean_y)/std_y
    ################# design the model ###################

    seed_torch(1029)
    net = MLP(X.shape[1], 0.1)

    net.apply(init_weights)
    print('CRPS_min: {}, RS_min: {}'.format(CRPS_min, RS_min))

    net_regr = PhysinformedNet_AR(
        module=net,
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=128,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks_AH,
        optimizer__weight_decay=np.exp(-8),
        beta = beta,
        CRPS_min = CRPS_min,
        RS_min = RS_min,
        mean = mean_y,
        std = std_y,
        # d = d,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
    )
    X = torch.from_numpy(X).float()
    X_t = torch.from_numpy(X_t).float()
    # st()
    Y = np.vstack([y.T, y_real.T]).T
    Y = torch.from_numpy(Y).float()

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


def train_std_GRU(X, X_t, y, y_real, delay, Dst_sel, \
    idx_storm, device, pred='gru', train=True):

    callname = 'Res/params_std_GRU_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        pred+'.pt'

    hidden_size = 32
    output_size = 1
    input_size = X.shape[2]
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=5),
                   ProgressBar()]

    max_X = X.max(axis = 0)
    min_X = X.min(axis = 0)

    mean_y = y_real.mean()
    std_y = y_real.std()

    mean_X = np.mean(X.reshape(-1, X.shape[2]), axis=0)
    std_X = np.std(X.reshape(-1, X.shape[2]), axis=0)
    
    X = (X - mean_X)/std_X
    X_t = (X_t - mean_X)/std_X

    # X = (X-min_X)/(max_X-min_X)
    # X_t = (X_t-min_X)/(max_X-min_X)
    beta, CRPS_min, RS_min = est_beta(X, y, y_real)

    ################# design the model ###################

    seed_torch(1029)
    net = lstm_reg(input_size,
                   hidden_size,
                   num_layers=2,
                   output_size=output_size)
    
    net.apply(init_weights)
    print('CRPS_min: {}, RS_min: {}'.format(CRPS_min, RS_min))

    net_regr = PhysinformedNet_AR_2D(
        module=net,
        max_epochs=100,
        lr=3e-3,
        train_split = ValidSplit(5),
        batch_size=128,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks_AH,
        optimizer__weight_decay=np.exp(-8),
        beta = beta,
        CRPS_min = CRPS_min,
        RS_min = RS_min,
        mean = mean_y,
        std = std_y,
        # d = d,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
    )
    X = torch.from_numpy(X).float()
    X_t = torch.from_numpy(X_t).float()
    
    Y = np.stack([y.squeeze(), y_real.squeeze()], axis=-1)
    Y = torch.from_numpy(Y).float()
    # st()
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


def QQ_plot(y_real_t, y_t, std_Y, figname):

    fig, ax = plt.subplots(3, 1, figsize=(10, 16))
    # import ipdb;ipdb.set_trace()

    # y_UQ = 
    Y_temp = (y_real_t.squeeze()-y_t.squeeze())/std_Y/np.sqrt(2)
    ax[0].hist(Y_temp, 50)
    # ax[0].set_xlabel('X')
    ax[0].set_ylabel('Number of samples')

    Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1/Y_temp.shape[0]
    F = (1+erf((y_real_t.squeeze()-y_t.squeeze())/std_Y/np.sqrt(2)))/2
    # st()
    ax[1].plot(Y_temp, F, '.')
    ax[1].set_ylabel('F = CDF(X)')
    # plt.show()

    stats.probplot(F, dist="norm", 
                      plot=pylab, 
                    #   ax = ax[2]
                      )
    ax[2].set_xlabel('Number of std')
    ax[2].set_ylabel('CDF(F)')
    ax[2].set_xlim(-2, 2)
    ax[2].set_ylim(0, 1)
    # plt.show()
    plt.savefig(figname, dpi=300)


def visualize(delay, date_idx, date_clu, y_pred_t, 
              y_real_t, y_t, y_Per_t, 
              std_Y, std_Y_per, name_clu, 
              color_clu, figname):

    fig, ax = plt.subplots(figsize=(16, 12))  
    idx = date_idx
    print('start date: {}'.format(date_clu[0]))
    print('end date: {}'.format(date_clu[-1]))

    ax.plot(date_clu, y_pred_t[idx, -1].squeeze(), \
        'y.-', label='GRU')
    ax.plot(date_clu, y_real_t[idx], 'r.-', label='Observation')
    for i, name in enumerate(name_clu):
        ax.plot(date_clu, y_t[i][idx], color_clu[i]+'.-', \
            label=name)
    if delay == 1:
        ax.plot(date_clu, y_Per_t[idx], 'w.-', \
            label='persistence')
    else:
        ax.plot(date_clu, y_Per_t[idx], 'w.-', \
            label='pred_'+str(delay-1))

    plt.xticks(rotation='vertical')
    # ax.set_ylim((-700, 100))
    ax.set_ylabel('Dst(nT)')
    ax.set_title('Delay:'+str(delay)+'h')
    mini = np.min([np.min(y_Per_t[idx]-std_Y_per[idx]),\
        np.min(y_pred_t[idx, -1].squeeze()-std_Y[idx])])
    maxi = np.max([np.max(y_Per_t[idx]+std_Y_per[idx]),\
        np.max(y_pred_t[idx, -1].squeeze()+std_Y[idx])])   
    ax.set_ylim(mini-20, maxi+20)

    if delay == 1:
        ax.fill_between(date_clu, 
                        y_Per_t[idx]-std_Y_per[idx], 
                        y_Per_t[idx]+std_Y_per[idx], 
                        # 'g',
                        color='gray',
                        label='Per_Uncertainty')
    else:
        ax.fill_between(date_clu, 
                        y_Per_t[idx]-std_Y_per[idx], 
                        y_Per_t[idx]+std_Y_per[idx], 
                        # 'g',
                        color='gray',
                        label='pred'+str(delay-1)+'_Uncertainty')    
    ax.fill_between(date_clu, 
                    y_pred_t[idx, -1].squeeze()-std_Y[idx], 
                    y_pred_t[idx, -1].squeeze()+std_Y[idx], 
                    label='GRU_Uncertainty')
    ax.legend()
    fig.savefig(figname, dpi=300)
