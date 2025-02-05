import numpy as np
import tqdm
from subprocess import check_output
import re
import h5py
import argparse
from ipdb import set_trace as st

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument("-length_min", type=int, default=0,
               help='how many events')
p.add_argument("-length_max", type=int, default=60,
               help='how many events')
p.add_argument("-delay_max", type=int, default=6,
               help='max delay')
p.add_argument("-delay_min", type=int, default=0,
               help='min delay')

########### arguments for Main.py ################
p.add_argument("-Omni_data", type=str,
               default='Data/Omni_data.pkl',
            #    default='Data/all_20211021-20211111.pkl',
               help='Omni file')
p.add_argument('-var_idx', nargs='+', type=int,
               default=[0, 1, 2, 3, 4, 5, 6, 7],
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
               help="if add the y_pred into dDst model")
p.add_argument("-ratio", type=float, default=1.0,
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
               help="Do not remove the best 10 percent \
                   during each iteration")
args = p.parse_args()

leng = 1
# n_combinations = np.arange(0, 60)
n_combinations = np.arange(args.length_min, args.length_max)
# n_combinations = np.arange(27, args.length_max)
# n_combinations = [i for i in args.use_case]
# import ipdb;ipdb.set_trace()
n_len = len(n_combinations)

name = ['train', 'test', 'KF']

results_clu = np.zeros([n_len, 2])
Per_clu = np.zeros([n_len, 2])

for delay in range(args.delay_min, args.delay_max):
    RMSE_clu = np.zeros([3, 4])

    for i in tqdm.tqdm(n_combinations):

        cmd_share = ['-model {}'.format(args.model),
                    '-Omni_data {}'.format(args.Omni_data),
                    '-ratio {}'.format(args.ratio),
                    '-train_per {}'.format(args.train_per),
                    '-smooth_width {}'.format(args.smooth_width),
                    '-DA_method {}'.format(args.DA_method[0]),
                    '-Dst_sel {}'.format(args.Dst_sel),
                    '-delay {}'.format(delay+1),
                    '-std_method {}'.format(args.std_method),
                    '-device {}'.format(args.device),
                    '-storm_idx {}'.format(i)]
        if args.boost_num > 0:
            cmd_head = ['python3 main_boost_multi.py',
                        '-boost_num {}'.format(args.boost_num),
                        '-DA_num {}'.format(args.DA_num),
                        '-boost_method {}'.format(args.boost_method),
                        '-criteria {}'.format(args.criteria)]
            filename = 'Res/'+str(args.boost_num)+\
                '/'+str(args.ratio)+\
                '/Uncertainty_'+\
                str(delay+1)+'-' +\
                str(args.Dst_sel)+'-'+\
                args.criteria+'.h5'
        else:
            cmd_head = ['python3 main.py']
            filename = 'Res/'+\
                'Uncertainty_'+\
                str(delay+1)+'-' +\
                str(args.Dst_sel)+'-'+'.h5'

        cmd = ' '.join(cmd_head+cmd_share)
        if args.iter_flag:
            cmd = cmd + ' -iter_flag'
        if args.real_flag:
            cmd = cmd + ' -real_flag'
        if args.dst_flag:
            cmd = cmd + ' -dst_flag'
        if args.std_flag:
            cmd = cmd + ' -std_flag'
        if args.per_flag:
            cmd = cmd + ' -per_flag'
        if args.final_flag:
            cmd = cmd + ' -final_flag'
        if args.pred_flag:
            cmd = cmd + ' -pred_flag'
        if args.QQplot:
            cmd = cmd + ' -QQplot'
        if args.pred_plot:
            cmd = cmd + ' -pred_plot'
        if args.removal:
            cmd = cmd + ' -removal'
        # print(cmd)
        # st()

        try:
            out = check_output([cmd], shell=True)
            with h5py.File(filename, 'r') as f:
                RMSE = np.array(f['RMSE_clu_'+str(i)])
                # print('##########'+str(delay+1)+'hr, '+str(i)+'th event #########')
                # for j in range(3):

                #     print('##########'+name[j]+'################')
                #     print('RMSE is {}'.format(round(RMSE[j, 0],2)))
                #     print('RMSE in strong/mild/quiet is {}/{}/{}'.format(
                #         round(RMSE[j, 1],2),
                #         round(RMSE[j, 2],2),
                #         round(RMSE[j, 3],2)
                #         ))
                f.close()
            print('{} th sample done!'.format(i))
        except:
            print('{} th sample is missing!'.format(i))

        RMSE_clu += RMSE/n_len
    RMSE = RMSE_clu

    # print('##########'+str(delay+1)+'hr, #########')
    for j in range(2,3):

        print('##########'+name[j]+'################')
        # print('RMSE is {}'.format(round(RMSE[j, 0],2)))
        print('RMSE in {}hr all/strong/mild/quiet is {}/{}/{}/{}'.format(
            str(delay+1),
            round(RMSE[j, 0],2),
            round(RMSE[j, 1],2),
            round(RMSE[j, 2],2),
            round(RMSE[j, 3],2)
                ))
    
