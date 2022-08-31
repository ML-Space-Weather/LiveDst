from cProfile import label
import pandas as pd
import numpy as np
import h5py
from multiprocessing import cpu_count, Pool
import time
from tqdm import tqdm
import datetime as dt
import os
from random import shuffle
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib import rc

from nets import CNN_1D, lstm_reg, lstm_gp, my_weight_rmse_CB
from nets import cdf_AH, norm_cdf
from nets import seed_torch, init_weights, PhysinformedNet, my_custom_loss_func

import torch
from torch.optim import Adam, AdamW, RMSprop
import torch.multiprocessing as mp

from skorch.dataset import CVSplit, ValidSplit
from skorch.callbacks import ProgressBar, Checkpoint
from skorch.callbacks import EarlyStopping, WarmRestartLR, LRScheduler
import sklearn
from sklearn.metrics import make_scorer
from scipy.ndimage import shift

from funs_Syn import train_Dst_boost, train_std_GRU_boost
from funs_Syn import train_std_boost


from ipdb import set_trace as st

import argparse

# argument

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument("-delay", type=int, default=1,
               help='predict hours')
p.add_argument('-var_idx', nargs='+', type=int,
               default=[0, 1, 2, 3, 4, 5, 6],
               help='Indices of the variables to use')
p.add_argument("-Dst_sel", type=int, default=-100,
               help='select peak for maximum Dst')
p.add_argument("-device", type=int, default=0,
               help='which GPU to use')
p.add_argument("-boost_num", type=int, default=5,
               help='number of boost iteration')
p.add_argument("-syn_num", type=int, default=2,
               help='number of syn data (e.g., 2 is 2003 Halloween)')
p.add_argument("-ratio", type=float, default=1.1,
               help='stretch ratio')
p.add_argument("-DA_num", type=int, default=5,
               help='number of boost iteration')
p.add_argument('-storm_idx', type=int, nargs='+',
               default=[33],
               help='which storm used for test')
p.add_argument("-boost_method", type=str, default='linear',
               choices=['linear', 'max'],
               help="which boost method is used")
p.add_argument("-criteria", type=str, 
               default='resi_std',
               choices=['resi_std', 'std', 'resi', 'diff_std'],
               help="which criteria for the boost method;\
                    resi_std means residuals/std")
args = p.parse_args()

######################## configuration ####################
delay = args.delay
Dst_sel = args.Dst_sel
storm_idx = args.storm_idx
img_format = 'jpg'
boost_method = args.boost_method
if args.device >= 10:
    device = torch.device("cpu")
else:
    # device = torch.device("cuda:"+str(args.device)) 
    device = torch.device("cuda:0") 
print(device)
ratio = args.ratio
var_idx = args.var_idx
boost_num = args.boost_num
syn_num = args.syn_num
criteria = args.criteria

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

filename_load = '/media/faraday/andong/SW_synthetic/preprocess/'+\
    'SW_Storm_'+\
    str(syn_num)+\
    '_sb_10.mat'

filename_save = '/media/faraday/andong/SW_synthetic/Res/'+\
    'SW_Storm_'+\
    str(syn_num)+\
    '_sb_10.mat'

names = ['N',
         'V',
         'B_norm',
         'Bz',
         't',
         'c',
         'Dst']

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

rc('font', **font)

################# data load ###################

data = sio.loadmat(filename_load, squeeze_me=True)

vari_list = ['Bx_synth', 
             'By_synth',
             'Bz_synth',
             'n_synth',
             'V_synth']


date_sta = data['last_orig_time']
date = data['date']
vari = data['vari']

data_clu = []
for vari_t in vari_list:
    data_t = pd.DataFrame(data[vari_t])
    data_t.interpolate(inplace=True)

    data_clu.append(np.asarray(data_t))

real_t = pd.DataFrame(vari)
real_t.interpolate(inplace=True)
print(f'Missing value count {real_t.isna().sum()}/{len(real_t)}')

vari = np.asarray(vari)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(vari[:, -1])
# st()
# st()
Bx = data_clu[0]
By = data_clu[1]
Bz = data_clu[2]
N = data_clu[3]
V = data_clu[4]

B_norm = np.sqrt(Bx**2 + By**2)
# Fill missing values

date_limit = dt.datetime(int(date_sta[0]),
                         int(date_sta[1]),
                         int(date_sta[2]),
                         int(date_sta[3]),
                         int(date_sta[4]),
                        )

date_end = dt.datetime(int(date[-1, 0]),
                       int(date[-1, 1]),
                       int(date[-1, 2]),
                       int(date[-1, 3]),
                       int(date[-1, 4]),
                       )

date_clu = []
date_plot = []
for i, date_tt in tqdm(enumerate(date)):
            
    t = dt.datetime(int(date_tt[0]),
                    int(date_tt[1]),
                    int(date_tt[2]),
                    int(date_tt[3]),
                    int(date_tt[4]),
                    )
    if t == date_limit:
        idx_sel = i
    if (t > date_limit) & (t <= date_end):
    # if (t > date_limit) & (t <= date_limit+dt.timedelta(hours=24*2)):
    # if (t > date_limit-dt.timedelta(hours=6)) & (t < date_limit+dt.timedelta(hours=24*3)):
    # if (t > date_limit-dt.timedelta(hours=6)):
        date_plot.append(t) 
    date_clu.append(t) 

################# SH variables ###################

DOY = np.asarray([date.timetuple().tm_yday for date in date_clu])
year = np.asarray([date.year for date in date_clu])
month = np.asarray([date.month for date in date_clu])
dom = np.asarray([date.day for date in date_clu])
UTC = np.asarray([date.hour for date in date_clu])
date_clu = np.vstack([year, month, dom, UTC]).T
t_year = 23.4*np.cos((DOY-172)*2*np.pi/365.25)
t_day = 11.2*np.cos((UTC-16.72)*2*np.pi/24)
t = t_year+t_day
theta_c = np.arctan2(By, Bz)

print('max of t_year/day is {} {}'.format(t_year.max(), t_day.max()))

SH_vari = np.vstack([np.expand_dims(np.tile(t[idx_sel+1:], (10000, 1)), 0),
                     np.expand_dims(np.sin(theta_c), 0),
]).T

X = vari
X_real = np.tile(np.expand_dims(X, 1), (1, 10000, 1))

print('################### generating X ###############')
# print(B_norm.shape)
# print(X_real.shape)

X_syn = np.vstack([
    np.expand_dims(N, 0), 
    np.expand_dims(V, 0), 
    np.expand_dims(B_norm, 0), 
    np.expand_dims(Bz, 0), 
    SH_vari.T, 
    np.expand_dims(X_real[idx_sel+1:, :, -1].T, 0), 
    ]).T


X_syn = np.swapaxes(np.vstack([X_real[idx_sel-5:idx_sel+1], X_syn]), 0, 1)
X_real = np.swapaxes(X_real, 0, 1)

X_syn = np.vstack([X_real[:2, idx_sel-5:], X_syn])

################# load training set ###################

with h5py.File('Data/data_'+str(delay)+
               '_'+str(Dst_sel)+'.h5', 'r') as f:

    idx = np.arange(np.array(f['num']))
    # shuffle(idx) # shuffle storm events
    idx = list(idx)
    # idx.remove(storm_idx[0])

    # print(f.keys())
    print(f['X_DL_{}'.format(idx[0])].shape)
    print(f['X_DL_{}'.format(idx[1])].shape)
    # st()
    # train
    X_train = np.array(f['X_DL_{}'.format(idx[0])][:, :, :])
    Y_train = np.array(f['Y_DL_{}'.format(idx[0])])      
    Dst_Per = np.array(f['Dst_Per{}'.format(idx[0])])        

    for i in tqdm(range(1, len(idx))):
        # import ipdb;ipdb.set_trace()
        # train_idx_clu.append(len(Y_train))
        X_train = np.vstack([X_train, \
            np.array(f['X_DL_{}'.format(idx[i])][:, :, :])
            ])
        Y_train = np.vstack([Y_train, \
            np.array(f['Y_DL_{}'.format(idx[i])])
            ])
        Dst_Per = np.hstack((Dst_Per, \
            np.array(f['Dst_Per{}'.format(idx[i])])
            ))
    # train_idx_clu.append(len(Y_train))

    # test
    X_test = np.array(f['X_DL_{}'.format(storm_idx[0])][:, :, :])
    Y_test = np.array(f['Y_DL_{}'.format(storm_idx[0])])
    Dst_Per_t = np.array(f['Dst_Per{}'.format(storm_idx[0])])
    date_test = np.array(f['date_DL_{}'.format(storm_idx[0])])

    for i in tqdm(range(1, len(storm_idx))):
    
        # test_idx_clu.append(len(Y_test))
        X_test = np.vstack([X_test,\
            np.array(f['X_DL_{}'.format(storm_idx[i])])
            ])
        Y_test = np.vstack([Y_test,\
            np.array(f['Y_DL_{}'.format(storm_idx[i])])
            ])
        Dst_Per_t = np.vstack([Dst_Per,\
            np.array(f['Dst_Per{}'.format(storm_idx[i])])
            ])
            
    # test_idx_clu.append(len(Y_test))
    
    f.close()


################# train ###################

res_clu = []
std_clu = []
X_syn_t = np.array(X_syn)

# for i in tqdm(range(6, 24*2)):
for i in tqdm(range(6, X_syn.shape[1])):

    y_clu = []
    std_clu = []

    std_Y_per = train_std_boost(
                    X_train[:, -1, var_idx], 
                    X_syn_t[:, i, var_idx], 
                    X_syn_t[:, i, -1], 
                    X_syn_t[:, i, -1], 
                    delay, Dst_sel, 
                    ratio, 0, boost_num, storm_idx[0], 
                    device, 
                    pred='per', 
                    criteria=criteria,
                    # train=std_model,
                    train=False
                    )

    y_clu = np.expand_dims(X_syn_t[:, i, -1], 0)
    std_clu = np.expand_dims(std_Y_per, 0)

    # st()
    
    for iter_boost in range(boost_num):

        # st()

        pred = train_Dst_boost(X_train[:, :, var_idx], 
                        X_train[:, :, -1], 
                        X_syn_t[:, i-6:i, var_idx], 
                        delay, Dst_sel, ratio, 
                        iter_boost, boost_num, 
                        # 1, 2,   
                        storm_idx[0], 
                        device, 
                        criteria, 
                        False)

        # st()

        std = train_std_GRU_boost(X_train[:, :, var_idx], 
                                X_syn_t[:, :, var_idx], 
                                pred, X_syn_t[:, i-6:i, -1], 
                                pred, X_syn_t[:, i-6:i, -1], 
                                delay, 
                                Dst_sel, 
                                ratio, iter_boost, 
                                boost_num, 
                                storm_idx[0], 
                                device, 
                                pred='gru', 
                                criteria=criteria, 
                                # train=std_model, 
                                train=False
                                )
        # st()
        # if iter_boost == 0:
        #     y_clu = np.expand_dims(pred[:, -1].squeeze(), 0)
        #     std_clu = np.expand_dims(std, 0)
        # else:
        y_clu = np.vstack([y_clu, 
                        np.expand_dims(pred[:, -1].squeeze(), 0)])
        std_clu = np.vstack([std_clu, 
                                np.expand_dims(std, 0)])

    # st()
    sigma_clu = 1/(std_clu**2)

    if boost_method == 'linear':
        y_t = sigma_clu*y_clu.squeeze()/sigma_clu.sum(axis=0)
        pred_final = y_t.sum(axis=0)
    elif boost_method == 'max':
        # st()
        arg = np.argmax(sigma_clu, axis=0)
        pred_final = np.zeros(sigma_clu.shape[1])
        for j in range(arg.shape[0]):
            pred_final[j] = y_clu[arg[j], j]
    else:
        pred_final = y_clu[0]
    X_syn_t[:, i, -1] = pred_final
    # X_syn_t[:, i, -1] = pred_final
    
    res_clu.append(pred_final)

res = np.zeros([res_clu[0].shape[0], len(res_clu)])
for i in range(res.shape[1]):
    res[:, i] = res_clu[i]

print(res.shape)

y_real = shift(X_real[0, :, -1], 1, cval=0)
y_syn = np.tile(y_real, (10000, 1))

############################## save and plot ###################

# y_syn[:,idx_sel+1:] = res[2:]
# sio.savemat(filename_save, {"y_syn":y_syn})
sio.savemat(filename_save, {"X_syn":X_syn_t[1:]})

# fig, axs = plt.subplots(X_syn_t.shape[2], 1, figsize=(16, 32))
# for idx, ax in enumerate(axs):
#     for i in range(10):
#         if i == 0:
#             ax.plot(X_syn_t[i*100, :, idx], 'o-', label='real')
#         else:
#             ax.plot(X_syn_t[i*100, :, idx], '-', label='syn_'+str(i))
#     ax.set_ylabel(names[idx])
#     ax.legend()

# # ax.plot(date_plot[1:42], y_real[idx_sel+1:idx_sel+42], 'o-',label='real')

# # for i in range(10):
# #     ax.plot(date_plot[1:42], y_syn[i*100, idx_sel+1:idx_sel+42], label='syn'+str(i+1))
#     # ax.plot(date_plot[1:42], y_syn[i*100, idx_sel+1:idx_sel+42], label='syn'+str(i+1))
# # plt.legend()
# ax.set_xlabel('Date')
# # ax.set_ylabel('Dst (nT)')
# plt.savefig('Figs/syn1.png')