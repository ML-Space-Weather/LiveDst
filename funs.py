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

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    # st()
    gpu_df = pd.read_csv(StringIO(u"".join(gpu_stats)),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx

def find_gpus(nums=6):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_free_gpus')
    with open('tmp_free_gpus', 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx,int(x.split()[2]))
                              for idx,x in enumerate(frees) ]
    
    idx_freeMemory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    usingGPUs = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freeMemory_pair[:nums] ]
    usingGPUs =  ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs

def setup(rank, world_size=3):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", 
                            rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def RMSE_dst(y_pred, y_real, Print=True):

    idx_strong = np.where(y_real<-100)[0]
    idx_mild = np.where((y_real<-50) & (y_real>=-100))[0]
    idx_quiet = np.where(y_real>-50)[0]
    RMSE = np.sqrt(np.mean((y_pred - y_real)**2))
    RMSE_strong = np.sqrt(np.mean((y_pred[idx_strong] - y_real[idx_strong])**2))
    RMSE_mild = np.sqrt(np.mean((y_pred[idx_mild] - y_real[idx_mild])**2))
    RMSE_quiet = np.sqrt(np.mean((y_pred[idx_quiet] - y_real[idx_quiet])**2))

    RMSE_clu = np.hstack((RMSE, 
                          RMSE_strong,
                          RMSE_mild,
                          RMSE_quiet))
    if Print==True:
        print('RMSE is {}'.format(round(RMSE,2)))
        print('RMSE in strong/mild/quiet is {}/{}/{}'.format(
            round(RMSE_strong,2),
            round(RMSE_mild,2),
            round(RMSE_quiet,2)
            ))
    # st()

    # return None
    return RMSE_clu

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


def storm_sel_omni(Omni_data, delay, Dst_sel, width):

    varis = ['N',    
             'V',    
             'BX_GSE',
             'BY_GSM',
             'BZ_GSM',
             'F10_INDEX',
             'DST'
             ]

    sta = ['HON', 'SJG', 'KAK', 'HER']
    
    # st()
    # Omni read
    df = pd.read_pickle(Omni_data)
    Omni_data = df[varis]
    Omni_date = df.index
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')

    # Fill missing values
    Omni_data.interpolate(inplace=True)
    Omni_data.interpolate(inplace=True)
    # Omni_data.dropna(inplace=True)
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')
    N = np.array(Omni_data['N'])
    V = np.array(Omni_data['V'])
    Bx = np.array(Omni_data['BX_GSE'])
    By = np.array(Omni_data['BY_GSM'])
    Bz = np.array(Omni_data['BZ_GSM'])
    F107 = np.array(Omni_data['F10_INDEX'])
    F107 = shift(F107, 24, cval=0)
    Dst = smooth(np.array(Omni_data['DST']), width)
    # Dst = stretch(Dst, ratio=ratio, width=Dst_sel)
    B_norm = np.sqrt(Bx**2+By**2)
    # B_norm = np.sqrt(Bx**2+By**2+Bz**2)
    
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
    # st()
    SH_vari = np.vstack([
                        #  np.sqrt(F107),
                         t,
                         np.sin(theta_c),
    ]).T

    ################## persistence ####################

    Y_Per = shift(Dst, delay, cval=np.NaN)
    error = np.sqrt(np.nanmean((Y_Per - Dst)**2))
    # print('RMSE of all persistence model:{}'.format(error))

    ################## NN ##############################
    X_NN = np.zeros([Dst.shape[0]-6-delay, 12])
    Y_NN = np.zeros(Dst.shape[0]-6-delay)
    date_NN = np.zeros([Dst.shape[0]-6-delay, 4])

    for i in tqdm(np.arange(5, Dst.shape[0]-delay-1)):

        X_NN[i-6] = np.hstack((N[i], V[i], B_norm[i], Bz[i],
                               SH_vari[i],
                               Dst[i-5:i+1]))
        Y_NN[i-6] = Dst[i+delay]
        date_NN[i-6] = date_clu[i+delay+1]
        

    ################## lstm/1D_CNN ##############################
    X_DL = np.zeros([Dst.shape[0]-6-delay, 6, 7])
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

    date_plot = []
    for i, date_tt in tqdm(enumerate(date_NN)):
                
        t = dt.datetime(int(date_tt[0]),
                        int(date_tt[1]),
                        int(date_tt[2]),
                        int(date_tt[3]),
                        )
        date_plot.append(t) 
    date_peak = []
    for i, date_tt in tqdm(enumerate(date_NN[peaks[idx]])):
                
        t = dt.datetime(int(date_tt[0]),
                        int(date_tt[1]),
                        int(date_tt[2]),
                        int(date_tt[3]),
                        )
        date_peak.append(t) 
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(date_plot[100:-100], Y_NN[100:-100], 'b-', 
            label='Dst index')
    ax.plot(date_peak, Y_NN[peaks[idx]], 'x',
            color='m', 
            # color='orange', 
            label='Peaks',
            markersize=12)
    # for peak_idx in peaks:
    #     ax.plot(date_plot[peak_idx], 
    #             Y_NN[peak_idx], 'mx', 
    #             # label='Dst index',
    #             markersize=12)
    ax.set_xlim(date_peak[0], date_peak[-1])
    ax.set_xlabel('Date')
    ax.set_ylabel('Dst (nT)')
    ax.legend()
    fig.savefig('Figs/storm_events.jpg', dpi=300)
    # st()


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


def storm_sel_realtime(Omni_data, delay, Dst_sel, width):

    varis = ['N',    
             'V',    
             'BX_GSM',
             'BY_GSM',
             'BZ_GSM',
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
    # Omni_data.dropna(inplace=True)
    
    print(f'Missing value count {Omni_data.isna().sum()}/{len(Omni_data)}')
    N = np.array(Omni_data['N'])
    V = np.array(Omni_data['V'])
    Bx = np.array(Omni_data['BX_GSM'])
    By = np.array(Omni_data['BY_GSM'])
    Bz = np.array(Omni_data['BZ_GSM'])
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
    # st()
    SH_vari = np.vstack([
                        #  np.sqrt(F107),
                         t,
                         np.sin(theta_c),
    ]).T

    ################## persistence ####################

    Y_Per = shift(Dst, delay, cval=np.NaN)
    error = np.sqrt(np.nanmean((Y_Per - Dst)**2))
    # print('RMSE of all persistence model:{}'.format(error))

    ################## NN ##############################
    X_NN = np.zeros([Dst.shape[0]-6-delay, 12])
    Y_NN = np.zeros(Dst.shape[0]-6-delay)
    date_NN = np.zeros([Dst.shape[0]-6-delay, 4])

    for i in tqdm(np.arange(5, Dst.shape[0]-delay-1)):

        X_NN[i-6] = np.hstack((N[i], V[i], B_norm[i], Bz[i],
                               SH_vari[i],
                               Dst[i-5:i+1]))
        Y_NN[i-6] = Dst[i+delay]
        date_NN[i-6] = date_clu[i+delay+1]
        

    ################## lstm/1D_CNN ##############################
    X_DL = np.zeros([Dst.shape[0]-6-delay, 6, 7])
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

    date_plot = []
    for i, date_tt in tqdm(enumerate(date_NN)):
                
        t = dt.datetime(int(date_tt[0]),
                        int(date_tt[1]),
                        int(date_tt[2]),
                        int(date_tt[3]),
                        )
        date_plot.append(t) 
    date_peak = []
    for i, date_tt in tqdm(enumerate(date_NN)):
                
        t = dt.datetime(int(date_tt[0]),
                        int(date_tt[1]),
                        int(date_tt[2]),
                        int(date_tt[3]),
                        )
        date_peak.append(t) 

    with h5py.File('Data/realtime_data_'+str(delay)+
                   '_'+str(Dst_sel)+'.h5', 'w') as f:
                
        print('{} & {} & {} & {}'.format(i, 
                                            Omni_date[0],
                                            Omni_date[-1],
                                            Y_NN.min()))

        f.create_dataset('X_NN',\
            data=X_NN[:Dst.shape[0]-12])
        f.create_dataset('Y_NN',\
            data=np.expand_dims(Y_NN[:Dst.shape[0]-12], axis=1))
        f.create_dataset('date_NN',\
            data=date_NN)
        f.create_dataset('X_DL',\
            data=X_DL[:Dst.shape[0]-12])
        f.create_dataset('Y_DL',\
            data=Y_DL[:Dst.shape[0]-12])
        f.create_dataset('Dst_Per',\
            data=Y_Per[:Dst.shape[0]-12])
        f.create_dataset('date_DL',\
            data=date_DL[:Dst.shape[0]-12])
    f.close()

    
def train_Dst(X, Y, X_t, delay, Dst_sel,
              idx_storm, device, train=False):

    callname = 'Res/'+\
        'params_std_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        '.pt'

    my_callbacks = [Checkpoint(f_params=callname),
                    LRScheduler(WarmRestartLR),
                    # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                    EarlyStopping(patience=10),
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
        train_split = ValidSplit(5),
        batch_size=1024,
        optimizer=torch.optim.AdamW,
        callbacks=my_callbacks,
        optimizer__weight_decay=np.exp(-4),
        # thres=Y_thres,
        thres=.5,
        # device='cuda',  # uncomment this to train with CUDA
        device=device,  # uncomment this to train with CUDA
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
    idx_storm, device, pred='mlp', train=True):

    callname = 'Res/'+\
        'params_std_'+\
        str(delay)+'-' +\
        str(Dst_sel)+'-'+\
        str(idx_storm)+'-'+\
        pred+'.pt'
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=10),
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
    idx_storm, 
    device, pred='gru', train=True):

    callname = 'Res/'+\
        'params_std_'+\
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
                   EarlyStopping(patience=10),
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


def QQ_plot(y_real, y_pred, std_y,
            y_real_t, y_pred_t, std_y_t, 
            valid_idx,
            figname):

    fig, ax = plt.subplots(1, 3, figsize=(24, 8))


    y_train = y_real[valid_idx:]
    y_valid = y_real[:valid_idx]

    #### train 

    y = y_pred[valid_idx:]

    Y_temp = (y_train.squeeze()-y.squeeze())\
        /std_y[valid_idx:]/np.sqrt(2)
    # print(valid_idx)
    # st()
    Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1\
        /Y_temp.shape[0]
    F = (1+erf((y_train.squeeze()-y.squeeze())\
        /std_y[valid_idx:]/np.sqrt(2)))/2

    sort_F = np.sort(F)
    y = np.arange(len(sort_F))/float(len(sort_F))
    ax[0].plot(sort_F, y)
    ax[0].plot(y, y)
    ax[0].set_ylabel('CDF(F)')
    ax[0].set_xlabel('Normal N(0, 1)')

    #### valid 
    y = y_pred[:valid_idx]

    Y_temp = (y_valid.squeeze()-y.squeeze())\
        /std_y[:valid_idx]/np.sqrt(2)

    Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1\
        /Y_temp.shape[0]
    F = (1+erf((y_valid.squeeze()-y.squeeze())\
        /std_y[:valid_idx]/np.sqrt(2)))/2

    sort_F = np.sort(F)
    y = np.arange(len(sort_F))/float(len(sort_F))
    ax[1].plot(sort_F, y)
    ax[1].plot(y, y)
    ax[1].set_xlabel('Normal N(0, 1)')

    #### test 

    Y_temp = (y_real_t.squeeze()-y_pred_t.squeeze())\
        /std_y_t/np.sqrt(2)

    Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1\
        /Y_temp.shape[0]
    F = (1+erf((y_real_t.squeeze()-y_pred_t.squeeze())\
        /std_y_t/np.sqrt(2)))/2

    sort_F = np.sort(F)
    y = np.arange(len(sort_F))/float(len(sort_F))
    ax[2].plot(sort_F, y)
    ax[2].plot(y, y)
    ax[2].set_xlabel('Normal N(0, 1)')

    ax[0].set_title('train')
    ax[1].set_title('valid')
    ax[2].set_title('test')

    # plt.show()
    plt.savefig(figname, dpi=300)


def QQ_plot_clu(y_real_t, y_t_clu, std_Yt_clu, 
                y_real, y_clu, std_Y_clu, 
                valid_idx,
                figname):

    r = np.zeros([y_t_clu.shape[0], 3])
    fig, ax = plt.subplots(y_t_clu.shape[0], 3, figsize=(20, 60))

    y_train = y_real[valid_idx:]
    y_valid = y_real[:valid_idx]

    for idx, y in enumerate(y_clu):
        y = y[valid_idx:]

        Y_temp = (y_train.squeeze()-y.squeeze())\
            /std_Y_clu[idx][valid_idx:]/np.sqrt(2)

        Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1\
            /Y_temp.shape[0]
        F = (1+erf((y_train.squeeze()-y.squeeze())\
            /std_Y_clu[idx][valid_idx:]/np.sqrt(2)))/2

        sort_F = np.sort(F)
        y = np.arange(len(sort_F))/float(len(sort_F))
        ax[idx, 0].plot(sort_F, y)
        ax[idx, 0].plot(y, y)

        ax[idx, 0].set_ylabel(str(idx)+':CDF(F)')
        ax[idx, 0].set_xlabel('')
        ax[idx, 0].set_title('')

    ax[-1, 0].set_xlabel('Number of std')

    for idx, y in enumerate(y_clu):
        y = y[:valid_idx]

        Y_temp = (y_valid.squeeze()-y.squeeze())\
            /std_Y_clu[idx][:valid_idx]/np.sqrt(2)

        Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1\
            /Y_temp.shape[0]
        F = (1+erf((y_valid.squeeze()-y.squeeze())\
            /std_Y_clu[idx][:valid_idx]/np.sqrt(2)))/2

        sort_F = np.sort(F)
        y = np.arange(len(sort_F))/float(len(sort_F))
        ax[idx, 1].plot(sort_F, y)
        ax[idx, 1].plot(y, y)
        ax[idx, 1].set_ylabel(str(idx)+':CDF(F)')
        ax[idx, 1].set_xlabel('')
        ax[idx, 1].set_title('')

    ax[-1, 1].set_xlabel('Number of std')

    for idx, y_t in enumerate(y_t_clu):
        Y_temp = (y_real_t.squeeze()-y_t.squeeze())\
            /std_Yt_clu[idx]/np.sqrt(2)

        Y_idx_sort = np.arange(0, 1, 1/Y_temp.shape[0])+1\
            /Y_temp.shape[0]
        F = (1+erf((y_real_t.squeeze()-y_t.squeeze())\
            /std_Yt_clu[idx]/np.sqrt(2)))/2

        sort_F = np.sort(F)
        y = np.arange(len(sort_F))/float(len(sort_F))
        ax[idx, 2].plot(sort_F, y)
        ax[idx, 2].plot(y, y)
        ax[idx, 2].set_ylabel('')
        ax[idx, 2].set_xlabel('')
        ax[idx, 2].set_title('')

    ax[-1, 2].set_xlabel('Number of std')

    ax[0, 0].set_title('train')
    ax[0, 1].set_title('valid')
    ax[0, 2].set_title('test')

    # plt.show()
    plt.savefig(figname, dpi=300)


def visualize(delay, date_idx, date_clu, y_pred_t, 
              y_real_t, y_t, y_Per_t, 
              std_Y, std_Y_per, name_clu, 
              color_clu, figname, idx_plot):

    fig, ax = plt.subplots(figsize=(20, 14))  
    idx = date_idx

    print('start date: {}'.format(date_clu[0]))
    print('end date: {}'.format(date_clu[-1]))

    date_plot = [date_clu[i] for i in idx_plot]

    ax.plot(date_clu, y_pred_t[idx].squeeze(), \
        'k.-', label='GRU')
    
    ax.plot(date_plot, \
        y_pred_t[idx[idx_plot]].squeeze(), \
        'mo', 
        markersize=20,
        label='sample for next model')
    # ax.fill_between(date_plot, 
    #                 y_pred_t[idx[idx_plot]].squeeze()-std_Y[idx[idx_plot]], 
    #                 y_pred_t[idx[idx_plot]].squeeze()+std_Y[idx[idx_plot]], 
    #                 interpolate=False, alpha=.3,
    #                 label='sample for next model')
    ax.plot(date_clu, y_real_t[idx], 'r.-', label='Observation')
    for i, name in enumerate(name_clu):
        ax.plot(date_clu, y_t[i][idx], color_clu[i]+'.-', \
            label=name)
    # if delay == 1:
    #     ax.plot(date_clu, y_Per_t[idx], 'w.-', \
    #         label='persistence')
    # else:
    #     ax.plot(date_clu, y_Per_t[idx], 'w.-', \
    #         label='pred_'+str(delay-1))

    plt.xticks(rotation='vertical')
    # ax.set_ylim((-700, 100))
    ax.set_ylabel('Dst(nT)')
    ax.set_title('Delay:'+str(delay)+'h')
    mini = np.min([np.min(y_Per_t[idx]-std_Y_per[idx]),\
        np.min(y_pred_t[idx].squeeze()-std_Y[idx])])
    maxi = np.max([np.max(y_Per_t[idx]+std_Y_per[idx]),\
        np.max(y_pred_t[idx].squeeze()+std_Y[idx])])   
    ax.set_ylim(mini-20, maxi+20)

    # if delay == 1:
    #     ax.fill_between(date_clu, 
    #                     y_Per_t[idx]-std_Y_per[idx], 
    #                     y_Per_t[idx]+std_Y_per[idx], 
    #                     # 'g',
    #                     color='gray',
    #                     label='Per_Uncertainty')
    # else:
    #     ax.fill_between(date_clu, 
    #                     y_Per_t[idx]-std_Y_per[idx], 
    #                     y_Per_t[idx]+std_Y_per[idx], 
    #                     # 'g',
    #                     color='gray',
    #                     label='pred'+str(delay-1)+'_Uncertainty') 
       
    ax.fill_between(date_clu, 
                    y_pred_t[idx].squeeze()-std_Y[idx], 
                    y_pred_t[idx].squeeze()+std_Y[idx], 
                    interpolate=True, alpha=.5,
                    label='GRU_Uncertainty')
    ax.legend()
    fig.savefig(figname, dpi=300)


def visualize_EN(delay, date_idx, date_clu, y_pred_t, 
              y_real_t, y_t, y_Per_t, 
              std_Y_clu, std_Y_per, name_clu, 
              color_clu, figname, idx_plot_clu):

    fig, ax = plt.subplots(len(idx_plot_clu),
                           figsize=(10, 60))  
    idx = date_idx

    print('start date: {}'.format(date_clu[0]))
    print('end date: {}'.format(date_clu[-1]))

    # st()

    for i, idx_plot in enumerate(idx_plot_clu):
        date_plot = [date_clu[i] for i in idx_plot]
        # st()
        ax[i].plot(date_clu, 
                    y_pred_t[i, idx].squeeze(), 
                    'k.-', 
                    label='ensemble_'+str(i))
        
        ax[i].plot(date_plot, \
            y_pred_t[i, idx[idx_plot]].squeeze(), \
            'mo', 
            markersize=10,
            label='samples for next model')
        # st()
        # ax[i].fill_between(date_clu, 
        #     y_pred_t[i, idx].squeeze()-std_Y_clu[i, idx], 
        #     y_pred_t[i, idx].squeeze()+std_Y_clu[i, idx], 
        #     where=std_Y_clu[i, idx]>np.median(std_Y_clu[i, idx]),
        #     interpolate=True, alpha=1,
        #     color="m",
        #     label='worst half samples')
        
        # st()
        # ax.fill_between(date_plot, 
        #                 y_pred_t[idx[idx_plot]].squeeze()-std_Y[idx[idx_plot]], 
        #                 y_pred_t[idx[idx_plot]].squeeze()+std_Y[idx[idx_plot]], 
        #                 interpolate=False, alpha=.3,
        #                 label='sample for next model')
        ax[i].plot(date_clu, y_real_t[idx], 
                           'r.-', label='Observation')
        # for n, name in enumerate(name_clu):
        #     ax[i].plot(date_clu, y_t[n][idx], 
        #                        color_clu[n]+'.-', 
        #                        label=name)
        # if delay == 1:
        #     ax.plot(date_clu, y_Per_t[idx], 'w.-', \
        #         label='persistence')
        # else:
        #     ax.plot(date_clu, y_Per_t[idx], 'w.-', \
        #         label='pred_'+str(delay-1))

        # ax.set_ylim((-700, 100))
        ax[i].set_ylabel('Dst(nT)')
        ax[i].set_title('ensemble:'+str(i))
        mini = np.min([np.min(y_Per_t[idx]-std_Y_per[idx]),\
            np.min(y_pred_t[i, idx].squeeze()-std_Y_clu[i, idx])])
        maxi = np.max([np.max(y_Per_t[idx]+std_Y_per[idx]),\
            np.max(y_pred_t[i, idx].squeeze()+std_Y_clu[i, idx])])   
        ax[i].set_ylim(mini-20, maxi+20)

        # if delay == 1:
        #     ax.fill_between(date_clu, 
        #                     y_Per_t[idx]-std_Y_per[idx], 
        #                     y_Per_t[idx]+std_Y_per[idx], 
        #                     # 'g',
        #                     color='gray',
        #                     label='Per_Uncertainty')
        # else:
        #     ax.fill_between(date_clu, 
        #                     y_Per_t[idx]-std_Y_per[idx], 
        #                     y_Per_t[idx]+std_Y_per[idx], 
        #                     # 'g',
        #                     color='gray',
        #                     label='pred'+str(delay-1)+'_Uncertainty') 
        
        ax[i].fill_between(date_clu, 
                        y_pred_t[i, idx].squeeze()-std_Y_clu[i, idx], 
                        y_pred_t[i, idx].squeeze()+std_Y_clu[i, idx], 
                        interpolate=True, alpha=.5,
                        label='Uncertainty')
        ax[i].legend(loc=4, fontsize='xx-small')
        if i != len(idx_plot_clu)-1:
            ax[i].get_xaxis().set_visible(False)
    plt.xticks(rotation='vertical')
    fig.savefig(figname, dpi=300)


def com_plot(date, date_idx, Dst, real, figname):

    fig, ax = plt.subplots(figsize=(20, 14))  
    for idx, Dst_t in enumerate(Dst):
        ax.plot(date, Dst_t[date_idx], \
            label='ensemble_'+str(idx))
    ax.plot(date, real[date_idx], 
            'o-', 
            markersize=10,
            label='Observation')
    ax.legend()
    plt.xticks(rotation='vertical')
    ax.set_ylabel('Dst(nT)')
    fig.savefig(figname, dpi=300)
    

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

    y_pred_t = net_regr.predict(X_t)#.reshape(-1, 1)
    y_pred_t = y_pred_t*Y_std+mean_Y

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
    
    my_callbacks_AH = [Checkpoint(f_params=callname),
                   LRScheduler(WarmRestartLR),
                   # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
                   EarlyStopping(patience=10),
                   ProgressBar()]

    max_X = X.max(axis = 0)
    min_X = X.min(axis = 0)

    if delay == 0:
        y = smooth(y, width=9)

    mean_y = y_real.mean()
    std_y = y_real.std()

    X = (X-min_X)/(max_X-min_X)
    X_t = (X_t-min_X)/(max_X-min_X)
    # st()
    beta, CRPS_min, RS_min = est_beta(X, y, y_real)
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
    beta, CRPS_min, RS_min = est_beta(X_t, y_t, y_real_t)
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