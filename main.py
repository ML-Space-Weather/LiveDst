from cProfile import label
import pandas as pd
import numpy as np
import h5py
from multiprocessing import cpu_count, Pool
import time
from tqdm import tqdm
import datetime as dt

import matplotlib.pyplot as plt
from matplotlib import rc

from nets import CNN_1D, lstm_reg, lstm_gp, my_weight_rmse_CB
from nets import cdf_AH, norm_cdf
from nets import seed_torch, init_weights, PhysinformedNet, my_custom_loss_func

import torch
from torch.optim import Adam, AdamW, RMSprop
from skorch.dataset import CVSplit, ValidSplit
from skorch.callbacks import ProgressBar, Checkpoint
from skorch.callbacks import EarlyStopping, WarmRestartLR, LRScheduler
import sklearn
from sklearn.metrics import make_scorer

from funs import smooth, stretch, est_beta, train_Dst, train_std_GRU
from funs import train_std, QQ_plot, visualize, storm_sel_omni, storm_sel_ACE

from ipdb import set_trace as st

import argparse

# argument

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument("-Omni_data", type=str,
               default='Data/Omni_data.pkl',
            #    default='Data/all_20211021-20211111.pkl',
               help='Omni file')
p.add_argument("-delay", type=int, default=1,
               help='predict hours')
p.add_argument('-var_idx', nargs='+', type=int,
               default=[0, 1, 2, 3, 4, 5, 6, 7],
               help='Indices of the variables to use')
p.add_argument("-Dst_sel", type=int, default=-100,
               help='select peak for maximum Dst')
p.add_argument("-smooth_width", type=int, default=0,
               help='width for smooth')
p.add_argument("-device", type=int, default=0,
               help='which GPU to use')
p.add_argument('-storm_idx', type=int, nargs='+',
               default=[33],
               help='which storm used for test')
p.add_argument("-model", type=str, default='GRU',
               choices=['GRU', 'KF'],
               help="which model result is used for AR")
p.add_argument("-std_method", type=str, default='MLP',
               choices=['GRU', 'MLP'],
               help="which method result is used to train std")
p.add_argument("-DA_method", type=str, nargs='+', 
               default=['Linear', 'KF_std', 'KF_real'],
               choices=['Linear', 'KF_std', 'KF_real'],
               help="which data assimilation model is used")
p.add_argument("-pred_flag", action='store_true',
               help="if add the y_pred-y_real")
p.add_argument("-ratio", type=float, default=1.1,
               help='stretch ratio')
p.add_argument("-Dst_flag", action='store_true',
               help="True: retrain Dst model; \
                   default:use the pre-trained one")
p.add_argument("-std_flag", action='store_true',
               help="True: retrain dDst model; \
                   default:use the pre-trained one")
p.add_argument("-iter_flag", action='store_true',
               help="True: use historical pred to replace persist; \
                   default:use the persistence model")
p.add_argument("-QQplot", action='store_true',
               help="Q-Q plot")
p.add_argument("-pred_plot", action='store_true',
               help="visualize predictions and DA results")
args = p.parse_args()

######################## configuration ####################
delay = args.delay
Dst_sel = args.Dst_sel
storm_idx = args.storm_idx
img_format = 'jpg'
width = args.smooth_width
std_method = args.std_method
DA_method = args.DA_method
device = torch.device("cuda:"+str(args.device)) 
pred = args.model # not used
ratio = args.ratio
Omni_data = args.Omni_data # not used
vari = args.var_idx

pred_flag = args.pred_flag
Dst_model = args.Dst_flag
std_model = args.std_flag
iter_mode = args.iter_flag
qq_plot = args.QQplot
visual_flag = args.pred_plot

filename_load = 'Res/Uncertainty_'+\
    str(delay-1)+'-' +\
    str(Dst_sel)+'-'+'.h5'

filename_save = 'Res/Uncertainty_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+'.h5'

figname_QQ = 'Figs/calibrate_UQ2_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+\
    str(storm_idx[0])+'-'+\
    pred+'.'+img_format 

figname_pred = 'Figs/predict_UQ2_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+\
    str(storm_idx[0])+'.'+img_format  

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

rc('font', **font)

######################## storm select ####################

if width == 0:
    pass
else:
    # meanwhile, smooth Dst
    storm_sel_omni(Omni_data, delay, Dst_sel, width) 
    # storm_sel_ACE(Omni_data, delay, Dst_sel, width, 60) 

######################## model Dst ####################

test_idx_clu = [0]
train_idx_clu = [0]

score = make_scorer(my_custom_loss_func,
                    greater_is_better=False)

with h5py.File('Data/data_'+str(delay)+
               '_'+str(Dst_sel)+'.h5', 'r') as f:

    idx = list(range(np.array(f['num'])))

    # print(f.keys())
    print(f['X_DL_{}'.format(idx[0])].shape)
    print(f['X_DL_{}'.format(idx[1])].shape)
    # st()
    # train
    X_train = np.array(f['X_DL_{}'.format(idx[0])][:, :, vari])
    Y_train = np.array(f['Y_DL_{}'.format(idx[0])])      
    Dst_Per = np.array(f['Dst_Per{}'.format(idx[0])])        

    for i in tqdm(range(1, len(idx))):
        # import ipdb;ipdb.set_trace()
        train_idx_clu.append(len(Y_train))
        X_train = np.vstack([X_train, \
            np.array(f['X_DL_{}'.format(idx[i])][:, :, vari])
            ])
        Y_train = np.vstack([Y_train, \
            np.array(f['Y_DL_{}'.format(idx[i])])
            ])
        Dst_Per = np.hstack((Dst_Per, \
            np.array(f['Dst_Per{}'.format(idx[i])])
            ))
    train_idx_clu.append(len(Y_train))

    # test
    X_test = np.array(f['X_DL_{}'.format(storm_idx[0])][:, :, vari])
    Y_test = np.array(f['Y_DL_{}'.format(storm_idx[0])])
    Dst_Per_t = np.array(f['Dst_Per{}'.format(storm_idx[0])])
    date_test = np.array(f['date_DL_{}'.format(storm_idx[0])])

    for i in tqdm(range(1, len(storm_idx))):
    
        test_idx_clu.append(len(Y_test))
        X_test = np.vstack([X_test,\
            np.array(f['X_DL_{}'.format(storm_idx[i])])
            ])
        Y_test = np.vstack([Y_test,\
            np.array(f['Y_DL_{}'.format(storm_idx[i])])
            ])
        Dst_Per_t = np.vstack([Dst_Per,\
            np.array(f['Dst_Per{}'.format(storm_idx[i])])
            ])
            
    test_idx_clu.append(len(Y_test))
    
    f.close()

if (delay > 1) & iter_mode:
    with h5py.File(filename_load, 'r') as f:
        # st()
        Dst_Per = np.array(f['y'+str(storm_idx[0])]) 
        Dst_Per_t = np.array(f['y_t'+str(storm_idx[0])]) 
        std_Per = np.array(f['std'+str(storm_idx[0])]) 
        std_Per_t = np.array(f['std_t'+str(storm_idx[0])]) 
        f.close()

X = X_train
X[:, :, -6:] = stretch(X[:, :, -6:], ratio=ratio, thres=Dst_sel)
Y = stretch(Y_train, ratio=ratio, thres=Dst_sel)

# print(X[:, -1, :].max(axis=0))
# print(Y[:, -1].max())

# print(X_test[:3, -1, :].max(axis=0))
# print(Y_test[:10, -1])
# print(Dst_Per_t[:10])
# Y = Y_train
# st()

X_t = X_test
Y_t = Y_test  

# st()
y_Per = stretch(Dst_Per, ratio=ratio, thres=Dst_sel)
y_Per_t = Dst_Per_t

if storm_idx[0] == 27:
    date_idx = np.arange(264, 360)
else:
    date_idx = np.arange(6, Y_t.shape[0]-6)
'''
date_idx = np.arange(6, Y_t.shape[0]-6)
'''

############################ model 
y_pred = train_Dst(X, Y, X, delay, Dst_sel, Dst_model)
y_pred_t = train_Dst(X, Y, X_t, delay, Dst_sel, False)

print('max of y_pred_t {}'.format(y_pred_t.max()))
print('min of y_pred_t {}'.format(y_pred_t.min()))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y_pred_t[:100, -1].squeeze(), 'b.-', label='GRU')
ax.plot(Y_t[:100, -1].squeeze(), 'm.-', label='real')
fig.savefig('Figs/test.jpg')
plt.close()
# st()
######################## model dDst ####################

date_clu = []
for i, date_tt in tqdm(enumerate(date_test[date_idx])):
            
    t = dt.datetime(int(date_tt[0]),
                    int(date_tt[1]),
                    int(date_tt[2]),
                    int(date_tt[3]),
                    )
    date_clu.append(t) 
    
y_real = Y_train[:, -1].squeeze()
y_real_t = Y_t[:, -1].squeeze()

# preprocess
y = y_pred[:, -1].squeeze()
y_t = y_pred_t[:, -1].squeeze()
x = X[:, -1, :].reshape([X.shape[0], -1]).squeeze()
x_t = X_t[:, -1, :].reshape([X_t.shape[0], -1]).squeeze()
# x = X[:, -1, :].squeeze()
# x_t = X_t[:, -1, :].squeeze()

if pred_flag:
    x = np.vstack([x.T, y.T-y_real.T]).T
    x_t = np.vstack([x_t.T, y_t.T-y_real_t.T]).T

# dd = y - y_real
# print(dd.shape)
# print('d_max is: {}'.format(dd.max()))
# print('d_min is: {}'.format(dd.min()))

if std_method == 'MLP':
    std_Y = train_std(x, x_t, y, y_real, delay, Dst_sel, 
                    storm_idx[0], device,
                    pred='gru', 
                    train=std_model,
                    #   train=False
                    )

elif std_method == 'GRU':
    std_Y = train_std_GRU(X, X_t, y_pred, Y_train, delay, 
                          Dst_sel, 
                          storm_idx[0], 
                          device,
                          pred='gru', 
                          train=std_model,
                        #   train=False
                          )

std_Y_per = train_std(x, x_t, y_Per, y_real, delay, Dst_sel, 
                    storm_idx[0], device, 
                    pred='per', 
                    train=std_model,
                    # train=False
                    )

# print(std_Y_per.max())
# print(std_Y_per.min())

# st()

if qq_plot:
    QQ_plot(y_real_t, y_t, std_Y, figname_QQ)

########################### KF part ############################

y_t_clu = []
name_clu = []
color_clu = []

if 'Linear' in DA_method:

    ########################### Linear estimator part 
    y_tr = y_pred[:, -1].squeeze()
    y_t = y_pred_t[:, -1].squeeze()

    std_Y_per_train = train_std(x, x, y_Per, y_real, 
                    delay, Dst_sel, 
                    storm_idx[0], 
                    device,
                    pred='per', 
                    train=False)
    std_Y_train = train_std(x, x, y, y_real, delay, Dst_sel, 
                    storm_idx[0],
                    device,
                    pred='gru', 
                    train=False)

    sigma_per_train = std_Y_per_train**2
    sigma_GRU_train = std_Y_train**2
    sigma_per = std_Y_per**2
    sigma_GRU = std_Y**2

    y_tr = sigma_per_train/(sigma_per_train+sigma_GRU_train)*y_tr+\
        sigma_GRU_train/(sigma_per_train+sigma_GRU_train)*y_Per

    y_t = sigma_per/(sigma_per+sigma_GRU)*y_t+\
        sigma_GRU/(sigma_per+sigma_GRU)*y_Per_t

    std_train = 1/(1/sigma_per_train + 1/sigma_GRU_train) # need to modify
    std_test = 1/(1/sigma_per + 1/sigma_GRU) # need to modify

    y_t_clu.append(y_t)
    name_clu.append('Linear')
    color_clu.append('g')

if 'KF_std' in DA_method:

    ################# with std ###########
    y_KF1 = np.array(y_pred_t[:, -1])
    P = 1/21
    Q = 1/16
    R = 1/16

    for idx, y in tqdm(enumerate(y_Per_t)):
        if y > Dst_sel:
            continue
        P = P+Q
        Resi = y - y_pred_t[idx, -1]
        K = P/(P+R)
        y_KF1[idx] = y_KF1[idx] + K*Resi
        P = (1-K)*P
        Q = std_Y[idx]**2
        R = std_Y_per[idx]**2
    
    y_t_clu.append(y_KF1)
    name_clu.append('KF_std')
    color_clu.append('m')

if 'KF_real' in DA_method:

    ################# with std ###########
    y_KF2 = np.array(y_pred_t[:, -1])
    P = 1/21
    Q = 1/16
    R = 1/16

    for idx, y in tqdm(enumerate(y_Per_t)):
        if y > Dst_sel:
            continue
        P = P+Q
        Resi = y - y_pred_t[idx, -1]
        K = P/(P+R)
        y_KF2[idx] = y_KF2[idx] + K*Resi
        P = (1-K)*P
        Q = Resi**2
        R = (y_real_t[idx-2*delay] - y_real_t[idx-delay])**2

    y_t_clu.append(y_KF2)
    name_clu.append('KF_real')
    color_clu.append('k')

# print('RMSE of training: {}'.format(np.mean((y-y_real)**2)))
# print('RMSE of test: {}'.format(np.mean((y_t-y_real_t)**2)))

with h5py.File(filename_save, 'a') as f:

    for w in ['y'+str(storm_idx[0]),
              'std'+str(storm_idx[0]),
              'y_t'+str(storm_idx[0]),
              'std_t'+str(storm_idx[0]),
              'y_gru'+str(storm_idx[0]),
              'y_gru_t'+str(storm_idx[0]),
                ]:
        if w in f:
            del f[w]
    # st()
    
    f.create_dataset('y'+str(storm_idx[0]),
                        data = y_tr.squeeze())
    f.create_dataset('std'+str(storm_idx[0]),
                        data = std_train.squeeze())
    f.create_dataset('y_t'+str(storm_idx[0]),
                        data = y_t.squeeze())
    f.create_dataset('std_t'+str(storm_idx[0]),
                        data = std_test.squeeze())
    f.create_dataset('y_gru'+str(storm_idx[0]),
                        data = y_pred[:, -1].squeeze())
    f.create_dataset('y_gru_t'+str(storm_idx[0]),
                        data = y_pred_t[:, -1].squeeze())
    f.close()

########################### visualize results ################

if visual_flag:
    visualize(delay, date_idx, date_clu, y_pred_t, 
              y_real_t, y_t_clu, y_Per_t, 
              std_Y, std_Y_per, name_clu, 
              color_clu, figname_pred)