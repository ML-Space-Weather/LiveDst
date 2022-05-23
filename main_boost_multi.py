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

from funs import smooth, storm_sel_realtime, stretch, est_beta, train_Dst, train_std_GRU
from funs import train_std, visualize, storm_sel_omni, storm_sel_realtime
from funs import train_Dst_boost, train_std_GRU_boost, train_std_boost
from funs import RMSE_dst, com_plot, visualize_EN, visualize
from funs import QQ_plot, QQ_plot_clu
from funs import setup

from ipdb import set_trace as st

import argparse

# argument

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument("-Omni_data", type=str,
            #    default='Data/Omni_data.pkl',
               default='Data/all_19990101-20170501.pkl',
               help='Omni file')
p.add_argument("-real_data", type=str,
            #    default='Data/Omni_data.pkl',
               default='Data/all_20220414-20220417.pkl',
               help='Omni file')
p.add_argument("-delay", type=int, default=1,
               help='predict hours')
p.add_argument('-var_idx', nargs='+', type=int,
               default=[0, 1, 2, 3],
               help='Indices of the variables to use')
p.add_argument("-Dst_sel", type=int, default=-100,
               help='select peak for maximum Dst')
p.add_argument("-smooth_width", type=int, default=0,
               help='width for smooth')
p.add_argument("-device", type=int, default=0,
               help='which GPU to use')
p.add_argument("-boost_num", type=int, default=5,
               help='number of boost iteration')
p.add_argument("-DA_num", type=int, default=5,
               help='number of boost iteration')
p.add_argument('-storm_idx', type=int, nargs='+',
               default=[33],
               help='which storm used for test')
p.add_argument("-model", type=str, default='GRU',
               choices=['GRU', 'KF'],
               help="which model result is used for AR")
p.add_argument("-boost_method", type=str, default='linear',
               choices=['linear', 'max'],
               help="which boost method is used")
p.add_argument("-std_method", type=str, default='MLP',
               choices=['GRU', 'MLP'],
               help="which method result is used to train std")
p.add_argument("-DA_method", type=str, nargs='+', 
               default=['Linear', 'KF_std', 'KF_real'],
               choices=['Linear', 'KF_std', 'KF_real'],
               help="which data assimilation model is used")
p.add_argument("-criteria", type=str, 
               default='resi_std',
               choices=['resi_std', 'std', 'resi', 'diff_std'],
               help="which criteria for the boost method;\
                    resi_std means residuals/std")
p.add_argument("-pred_flag", action='store_true',
               help="if add the y_pred")
p.add_argument("-ratio", type=float, default=1.1,
               help='stretch ratio')
p.add_argument("-train_per", type=float, default=5,
               help='How much percentage for training during boost')
p.add_argument("-real_flag", action='store_true',
               help="True: predict realtime data; \
                   default:predict postprocessed data")
p.add_argument("-dst_flag", action='store_true',
               help="True: retrain Dst model; \
                   default:use the pre-trained one")
p.add_argument("-std_flag", action='store_true',
               help="True: retrain dDst model; \
                   default:use the pre-trained one")
p.add_argument("-per_flag", action='store_true',
               help="True: retrain dPer model; \
                   default:use the pre-trained one")
p.add_argument("-final_flag", action='store_true',
               help="True: retrain final std model; \
                   default:use the pre-trained one")
p.add_argument("-iter_flag", action='store_true',
               help="True: use historical pred to replace persist; \
                   default:use the persistence model")
p.add_argument("-QQplot", action='store_true',
               help="Q-Q plot")
p.add_argument("-pred_plot", action='store_true',
               help="visualize predictions and DA results")
p.add_argument("-removal", action='store_true',
               help="remove the best 10 percent \
                   during each iteration")
args = p.parse_args()

######################## configuration ####################
delay = args.delay
Dst_sel = args.Dst_sel
storm_idx = args.storm_idx
img_format = 'jpg'
width = args.smooth_width
std_method = args.std_method
DA_method = args.DA_method
boost_method = args.boost_method
if args.device >= 10:
    device = torch.device("cpu")
else:
    # device = torch.device("cuda:"+str(args.device)) 
    device = torch.device("cuda:0") 
print(device)
pred = args.model # not used
ratio = args.ratio
train_per = args.train_per
Omni_data = args.Omni_data 
Real_data = args.real_data 
vari = args.var_idx
boost_num = args.boost_num
DA_num = args.DA_num
criteria = args.criteria

real_flag = args.real_flag
pred_flag = args.pred_flag
Dst_model = args.dst_flag
std_model = args.std_flag
per_model = args.per_flag
final_model = args.final_flag
iter_mode = args.iter_flag
qq_plot = args.QQplot
visual_flag = args.pred_plot
removal = args.removal

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

# world_size = 3
# mp.spawn(setup,
#     args=(world_size,),
#     nprocs=world_size,
#     join=True)

os.makedirs('Res/'+str(boost_num)+'/'+str(ratio)+'/', exist_ok=True)
os.makedirs('Figs/'+str(boost_num)+'/'+str(ratio)+'/', exist_ok=True)

filename_load = 'Res/'+str(boost_num)+\
    '/'+str(ratio)+\
    '/Uncertainty_'+\
    str(delay-1)+'-' +\
    str(Dst_sel)+'-'+\
    criteria+'.h5'

filename_save = 'Res/'+str(boost_num)+\
    '/'+str(ratio)+\
    '/Uncertainty_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+\
    criteria+'.h5'

figname_QQ = 'Figs/'+str(boost_num)+\
    '/'+str(ratio)+\
    '/calibrate_boost_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+\
    str(storm_idx[0])+'-'+\
    pred+\
    criteria+\
    '.'+img_format 

figname_pred = 'Figs/'+str(boost_num)+\
    '/'+str(ratio)+\
    '/predict_boost_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+\
    str(storm_idx[0])+\
    criteria+\
    '.'+img_format+'.'+img_format  

figname_EN = 'Figs/'+str(boost_num)+\
    '/'+str(ratio)+\
    '/predict_ensemble_'+\
    str(delay)+'-' +\
    str(Dst_sel)+'-'+\
    str(storm_idx[0])+\
    criteria+\
    '.'+img_format  

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

    if args.real_flag:
        storm_sel_realtime(Real_data, delay, Dst_sel, width)
######################## model Dst ####################

test_idx_clu = [0]
train_idx_clu = [0]

score = make_scorer(my_custom_loss_func,
                    greater_is_better=False)

with h5py.File('Data/data_'+str(delay)+
               '_'+str(Dst_sel)+'.h5', 'r') as f:

    idx = np.arange(np.array(f['num']))
    shuffle(idx) # shuffle storm events
    idx = list(idx)
    # idx.remove(storm_idx[0])

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

if args.real_flag:

    with h5py.File('Data/realtime_data_'+str(delay)+
               '_'+str(Dst_sel)+'.h5', 'r') as f:

        # test
        X_test = np.array(f['X_DL'][:, :, vari])
        Y_test = np.array(f['Y_DL'])
        Dst_Per_t = np.array(f['Dst_Per'])
        date_test = np.array(f['date_DL'])
                
    test_idx_clu.append(len(Y_test))
    
    f.close()

# st()
if (delay > 1) & iter_mode:
    with h5py.File(filename_load, 'r') as f:

        # plt.plot(Dst_Per_t[264:360], 'r.-', label='persistence')
        # st()
        Dst_Per = np.array(f['y'+str(storm_idx[0])]) 
        Dst_Per_t = np.array(f['y_t'+str(storm_idx[0])]) 
        Dst_Per = shift(Dst_Per, 1, cval=Dst_Per[0])
        Dst_Per_t = shift(Dst_Per_t, 1, cval=Dst_Per_t[0])
        # st()

        # plt.plot(Dst_Per_t[264:360], 'b.-', label='KF')
        # plt.plot(Y_test[264:360:, -1], 'g.-', label='real')
        # plt.legend()
        # plt.savefig('per_compare.jpg', dpi=300)
        # std_Per = np.array(f['std'+str(storm_idx[0])]) 
        # std_Per_t = np.array(f['std_t'+str(storm_idx[0])]) 
        f.close()

# st()
X = X_train
X[:, :, -6:] = stretch(X[:, :, -6:], ratio=ratio, thres=Dst_sel)
Y = stretch(Y_train, ratio=ratio, thres=Dst_sel)

print('shape of X_train: {}'.format(X_train.shape))
print('shape of X_test: {}'.format(X_test.shape))

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

if (storm_idx[0] == 32) & (real_flag==0):
    date_idx = np.arange(264, 360)
else:
    date_idx = np.arange(6, Y_t.shape[0]-6)
'''
date_idx = np.arange(6, Y_t.shape[0]-6)
'''
# st()

date_clu = []
for i, date_tt in tqdm(enumerate(date_test[date_idx])):
            
    t = dt.datetime(int(date_tt[0]),
                    int(date_tt[1]),
                    int(date_tt[2]),
                    int(date_tt[3]),
                    )
    date_clu.append(t) 
    
# st()

X_ori = X
Y_train_ori = Y_train
y_real = Y_train[:, -1].squeeze()
y_real_t = Y_t[:, -1].squeeze()
x = X[:, -1, :].reshape([X.shape[0], -1]).squeeze()
x_t = X_t[:, -1, :].reshape([X_t.shape[0], -1]).squeeze()
y_pred = train_Dst_boost(X, Y, X, delay, Dst_sel, 
                ratio, 0, boost_num, storm_idx[0], 
                device, 
                criteria,
                Dst_model)
y_pred_t = train_Dst_boost(X, Y, X_t, delay, Dst_sel, 
                ratio, 0, boost_num, storm_idx[0], 
                device, 
                criteria,
                False)

# st()
std_Y = train_std_boost(x, x, y_Per, 
                    y_real, delay, Dst_sel, 
                    ratio, 0, boost_num, storm_idx[0], 
                    device, 
                    pred='per', 
                    criteria=criteria,
                    train=per_model,
                    # train=False
                    )
std_Y_per = train_std_boost(x, x_t, y_Per, 
                    y_real, delay, Dst_sel, 
                    ratio, 0, boost_num, storm_idx[0], 
                    device, 
                    pred='per', 
                    criteria=criteria,
                    # train=std_model,
                    train=False
                    )

y_pred_save, y_pred_t_save, std_Y_save, std_Y_per_save = \
    y_pred, y_pred_t, std_Y, std_Y_per
for iter_boost in range(DA_num):
# for iter_boost in range(boost_num):
    n_sample = X.shape[0]

    print('boost no. {}'.format(iter_boost+1))
    print('num of samples left is {}'.format(n_sample))
    if iter_boost >= 0:
        # st()
        if criteria == 'resi_std':
            num_std = np.abs(y_real-y_pred[:, -1].squeeze())\
                /std_Y
        elif criteria == 'resi':
            num_std = np.abs(y_real-y_pred[:, -1].squeeze())
        elif criteria == 'std':
            num_std = std_Y
        elif criteria =='diff_std':
            num_std = np.abs(y_pred[:, -2].squeeze() - \
                y_pred[:, -1].squeeze())/std_Y
        std_Y_sort = np.sort(num_std)[::-1]
        idx_sort = np.argsort(num_std)[::-1]
        # st()
        if removal:
            X = X[idx_sort[:n_sample//10*9]]
            Y = Y[idx_sort[:n_sample//10*9]]
            y_real = y_real[idx_sort[:n_sample//10*9]]
            Y_train = Y_train[idx_sort[:n_sample//10*9]]
        else:
            X = X[idx_sort]
            Y = Y[idx_sort]
            y_real = y_real[idx_sort]
            Y_train = Y_train[idx_sort]
        ############################ model 
        # if iter_boost > 0:
        # st()
        y_pred = train_Dst_boost(X[:int(n_sample*train_per)//10], Y[:int(n_sample*train_per)//10], 
                        X, delay, Dst_sel, ratio, 
                        iter_boost, boost_num,
                        storm_idx[0], 
                        device, 
                        criteria,
                        Dst_model)
        y_pred_t = train_Dst_boost(X[:int(n_sample*train_per)//10], Y[:int(n_sample*train_per)//10], 
                            X_t, delay, Dst_sel, ratio, 
                            iter_boost, boost_num, 
                            storm_idx[0], 
                            device, 
                            criteria,
                            False)

    print('\n######### training set ###########')
    RMSE_dst(y_pred, y_real)
    print('\n######### test set ###########')
    RMSE_dst(y_pred_t, y_real_t)
    print('\n')
    ######################## model dDst ####################

    # preprocess
    y = y_pred[:, -1].squeeze()
    y_t = y_pred_t[:, -1].squeeze()
    x = X[:, -1, :].reshape([X.shape[0], -1]).squeeze()
    x_t = X_t[:, -1, :].reshape([X_t.shape[0], -1]).squeeze()

    if pred_flag:
        
        yy = Y_train[:, -2].squeeze() - Y_train[:, -1].squeeze()
        yy_t = Y_t[:, -2].squeeze() - Y_t[:, -1].squeeze()
        x = np.vstack([x.T, yy.T]).T
        x_t = np.vstack([x_t.T, yy_t.T]).T

    if std_method == 'MLP':
        # st()
        std_Y = train_std_boost(x, x, y, y_real, delay, Dst_sel, 
                        ratio, iter_boost, boost_num, 
                        storm_idx[0], device,
                        pred='gru', 
                        criteria=criteria,
                        train=std_model,
                        #   train=False
                        )
    # st()
    elif std_method == 'GRU':
        # st()
        std_Y = train_std_GRU_boost(X, X, 
                            y_pred, Y_train, 
                            y_pred_t, Y_test, 
                            delay, 
                            Dst_sel, 
                            ratio, iter_boost, 
                            boost_num,
                            storm_idx[0], 
                            device,
                            pred='gru', 
                            criteria=criteria,
                            train=std_model,
                            #   train=False
                            )

X = X_ori
Y_train = Y_train_ori
Y = stretch(Y_train, ratio=ratio, thres=Dst_sel)
y_real = Y_train[:, -1].squeeze()
y_pred = train_Dst_boost(X, Y, 
                X, delay, Dst_sel, ratio, 
                iter_boost, boost_num, 
                storm_idx[0], 
                device, 
                criteria,
                Dst_model)
y = y_pred[:, -1].squeeze()
x_ori = X[:, -1, :].reshape([X.shape[0], -1]).squeeze()
print('\n######### training set ###########')
RMSE_train = RMSE_dst(y_pred, y_real)
print('\n######### test set ###########')
RMSE_test = RMSE_dst(y_pred_t, y_real_t)
print('\n')

# st()

########################### KF part ############################

y_train_clu = []
y_test_clu = []
name_clu = []
color_clu = []
RMSE_opt = []

########################### Linear estimator part 

std_Y_per_train = train_std_boost(
                    x_ori, 
                    x_ori, 
                    y_Per, 
                    y_real, 
                    delay, Dst_sel, 
                    ratio, 0, boost_num,
                    storm_idx[0], 
                    device,
                    pred='per', 
                    criteria=criteria,
                    train=False)

std_Y_train_clu = np.expand_dims(std_Y_per_train, 0)
std_Y_test_clu = np.expand_dims(std_Y_per, 0)
y_tr_clu = y_Per
y_t_clu = y_Per_t
# st()
for iter_boost in range(DA_num):
# for iter_boost in range(boost_num):

    y_pred = train_Dst_boost(X[:int(n_sample*train_per)//10], 
                    Y[:int(n_sample*train_per)//10], 
                    X, delay, Dst_sel, ratio, 
                    iter_boost, boost_num, 
                    storm_idx[0], 
                    device,
                    criteria, 
                    False)
    y_pred_t = train_Dst_boost(X[:int(n_sample*train_per)//10], 
                    Y[:int(n_sample*train_per)//10], 
                    X_t, delay, Dst_sel, ratio, 
                    iter_boost, boost_num, 
                    storm_idx[0], 
                    device, 
                    criteria, 
                    False)

    tr_len = y_pred.shape[0]
    # y_tr = y_pred[:y_pred.shape[0]//5, -1].squeeze() # valid
    y_tr = y_pred[:, -1].squeeze() # train+valida
    y_t = y_pred_t[:, -1].squeeze()


    if std_method == 'GRU':
        std_Y_train = train_std_GRU_boost(X, X, y_pred, Y_train, y_pred_t, Y_test,
                            delay, 
                            Dst_sel, 
                            ratio, 
                            iter_boost, 
                            boost_num,
                            storm_idx[0], 
                            device,
                            pred='gru', 
                            criteria=criteria, 
                            train=False,
                            #   train=False
                            )

        std_Y = train_std_GRU_boost(X, X_t, y_pred, Y_train, 
                            y_pred_t, Y_test, 
                            delay, 
                            Dst_sel, 
                            ratio, 
                            iter_boost, 
                            boost_num,
                            storm_idx[0], 
                            device,
                            pred='gru', 
                            criteria=criteria, 
                            # train=std_model,
                            train=False
                            )
    else:
        std_Y_train = train_std_boost(x, x, y, y_real, delay, Dst_sel, 
                        ratio, iter_boost, 
                        iter_boost, 
                        boost_num,
                        storm_idx[0],
                        device,
                        pred='gru', 
                        criteria=criteria, 
                        train=False)

        std_Y = train_std_boost(x, x_t, y, y_real, delay, Dst_sel, 
                        ratio, iter_boost, 
                        iter_boost, 
                        boost_num,
                        storm_idx[0], device,
                        pred='gru', 
                        criteria=criteria, 
                        # train=std_model,
                        train=False
                        )

    # st()
    y_tr_clu = np.vstack([y_tr_clu, 
                            np.expand_dims(y_tr, 0)])
    y_t_clu = np.vstack([y_t_clu, np.expand_dims(y_t, 0)])
    std_Y_train_clu = np.vstack([std_Y_train_clu, 
                                    np.expand_dims(std_Y_train, 0)])
    std_Y_test_clu = np.vstack([std_Y_test_clu, 
                                    np.expand_dims(std_Y, 0)])

sigma_train_clu = 1/(std_Y_train_clu**2)
sigma_test_clu = 1/(std_Y_test_clu**2)

# fig, ax = plt.subplots(6, 2, figsize=(8, 24))
# for i in range(DA_num+1):
#     ax[i, 0].plot(np.abs(y_tr_clu[i]-y_real), sigma_train_clu[i], '.')
#     ax[i, 1].plot(np.abs(y_t_clu[i]-y_real_t), sigma_test_clu[i], '.')

#     ax[i, 0].set_ylabel('sigma')
#     ax[i, 0].set_title('train_'+str(i))
#     ax[i, 1].set_title('test_'+str(i))
# ax[i, 0].set_xlabel('abs(err)')
# ax[i, 1].set_xlabel('abs(err)')
# plt.savefig('test.png')

# sigma_train_clu = sigma_train_clu/sigma_train_clu.sum(axis=0)
# sigma_test_clu = sigma_test_clu/sigma_test_clu.sum(axis=0)
com_plot(date_clu, date_idx, 
            y_t_clu, y_real_t,
            figname_EN)
if qq_plot:
    r = QQ_plot_clu(y_real_t, y_t_clu, std_Y_test_clu, 
                y_real, y_tr_clu, std_Y_train_clu, 
                tr_len//5,
                figname_QQ)

y_t_clu_t = np.zeros(y_t_clu.shape[1])
for tr_num in range(1, DA_num+1):
    # st()
    y_tr = sigma_train_clu*y_tr_clu/sigma_train_clu[:tr_num+1].sum(axis=0)
    # y_t = sigma_test_clu*y_t_clu/sigma_test_clu[:tr_num+1].sum(axis=0)

    y_tr_clu_t = y_tr[:tr_num+1].sum(axis=0)
    # y_t_clu_t = y_t[:tr_num+1].sum(axis=0)

    if boost_method == 'max':
        idx = sigma_test_clu[:tr_num+1].argmin(axis=0)
        sigma_opt_clu = sigma_test_clu[:tr_num+1].min(axis=0)
        for i in np.arange(y_t_clu_t.shape[0]):
            y_t_clu_t[i] = y_t_clu[idx[i], i]

    elif boost_method == 'linear':
        y_t = sigma_test_clu*y_t_clu/sigma_test_clu[:tr_num+1].sum(axis=0)
        y_t_clu_t = y_t[:tr_num+1].sum(axis=0)

    RMSE_train = RMSE_dst(y_tr_clu_t[:tr_len//5], y_real[:tr_len//5], Print=False)
    RMSE_t = RMSE_dst(y_t_clu_t, y_real_t, Print=False)
    
    if tr_num == 1:
        RMSE_min = RMSE_dst(y_t_clu_t, y_real_t)
        RMSE_opt = RMSE_train[0]
        y_t_truth = y_t_clu_t
        y_tr_truth = y_tr_clu_t
        tr_num_truth = 1

    elif RMSE_train[0]<RMSE_opt:
        print('------------update---------------')
        print('RMSE_train in {} iteration is {}'
            .format(tr_num, RMSE_train[0]))
        RMSE_min = RMSE_dst(y_t_clu_t, y_real_t)
        y_tr_truth = y_tr_clu_t
        y_t_truth = y_t_clu_t
        tr_num_truth = tr_num
        RMSE_opt = RMSE_t[0]
        # st()

print('best DA_num is {}, whose RMSE is {}/{}/{}/{}'.format(
    tr_num_truth, 
    round(RMSE_min[0], 2),
    round(RMSE_min[1], 2),
    round(RMSE_min[2], 2),
    round(RMSE_min[3], 2),
))

# st()

std_train_final = train_std_boost(x_ori, x_ori, y_tr_truth, 
                    y_real, 
                    delay, Dst_sel, 
                    ratio, boost_num+1, boost_num,
                    storm_idx[0], 
                    device,
                    pred='final', 
                    criteria=criteria,
                    train=final_model)

std_test_final = train_std_boost(x_ori, x_t, y_tr_truth, 
                    y_real, 
                    delay, Dst_sel, 
                    ratio, boost_num+1, boost_num,
                    storm_idx[0], 
                    device,
                    pred='final', 
                    criteria=criteria,
                    train=False)


# std_train = 1/(1/sigma_per_train + 1/sigma_GRU_train) # need to modify
# std_test = 1/(1/sigma_per + 1/sigma_GRU) # need to modify

y_train_clu.append(y_tr_truth)
y_test_clu.append(y_t_truth)
name_clu.append('boost')
color_clu.append('g')

# print('RMSE of training: {}'.format(np.mean((y-y_real)**2)))
# print('RMSE of test: {}'.format(np.mean((y_t-y_real_t)**2)))
# st()
# RMSE_KF = RMSE_dst(y_t, y_real_t)
RMSE_final = RMSE_min

RMSE_clu = np.stack((RMSE_train, RMSE_test, RMSE_final))
# print(RMSE_clu.shape)
with h5py.File(filename_save, 'a') as f:

    for w in ['y'+str(storm_idx[0]),
              'std'+str(storm_idx[0]),
              'y_t'+str(storm_idx[0]),
              'std_t'+str(storm_idx[0]),
              'y_final'+str(storm_idx[0]),
              'y_final_t'+str(storm_idx[0]),
              'std_final'+str(storm_idx[0]),
              'std_final_t'+str(storm_idx[0]),
              'y_gru'+str(storm_idx[0]),
              'y_gru_t'+str(storm_idx[0]),
              'RMSE_clu_'+str(storm_idx[0]),
              'train_idx_'+str(storm_idx[0]),
                ]:
        if w in f:
            del f[w]
    # st()
    f.create_dataset('y'+str(storm_idx[0]),
                        data = y_real.squeeze())
    f.create_dataset('std'+str(storm_idx[0]),
                        data = std_Y_train_clu.squeeze())
    f.create_dataset('y_t'+str(storm_idx[0]),
                        data = y_real_t.squeeze())
    f.create_dataset('std_t'+str(storm_idx[0]),
                        data = std_Y_test_clu.squeeze())

    f.create_dataset('y_final'+str(storm_idx[0]),
                        data = y_tr_truth)
    f.create_dataset('y_final_t'+str(storm_idx[0]),
                        data = y_t_truth)
    f.create_dataset('std_final'+str(storm_idx[0]),
                        data = std_train_final)
    f.create_dataset('std_final_t'+str(storm_idx[0]),
                        data = std_test_final)

    f.create_dataset('y_gru'+str(storm_idx[0]),
                        data = y_pred_save[:, -1].squeeze())
    f.create_dataset('y_gru_t'+str(storm_idx[0]),
                        data = y_pred_t_save[:, -1].squeeze())

    f.create_dataset('RMSE_clu_'+str(storm_idx[0]),
                        data = RMSE_clu)
    f.create_dataset('train_idx_'+str(storm_idx[0]),
                        data = train_idx_clu)
    f.close()

########################### visualize results ################

# st()
idx_plot_clu = []
for idx in range(y_tr_clu.shape[0]):
    dd = np.abs(y_real_t[date_idx] - y_tr_clu[idx, date_idx])\
        /sigma_test_clu[idx, date_idx]
    idx_sort = np.argsort(dd)[::-1]
    idx_plot = idx_sort[:int(len(idx_sort)*train_per)//10].astype(int)
    dd_thres = dd[idx_plot[-1]]
    idx_plot_clu.append(idx_plot)

# st()
if visual_flag:
    visualize_EN(delay, date_idx, date_clu, y_t_clu, 
              y_real_t, y_test_clu, y_Per_t, 
              std_Y_test_clu, std_Y_per, name_clu, 
              color_clu, figname_pred, idx_plot_clu)
    
    # visualize(delay, date_idx, date_clu, y_t_clu, 
    #           y_real_t, y_test_clu, y_Per_t, 
    #           std_Y, std_Y_per, name_clu, 
    #           color_clu, figname_pred, idx_plot)