# Geomagnetic Storm Forecasting

Notebooks for a project with aim of predicting the Disturbance Storm-Time (Dst) 1-6 hours ahead using Gated Recurrent Units (GRU), Accurate abd Reliable Uncertainty Estimate (ACCRUE), and developed ensemble boost methods.

If you have any questions w.r.t the code, please contact andong.hu@colorado.edu or huan.winter@gmail.com.

## Overview

This project is conducted primarily in python that present theory alongside application. 

Two folders 'Figs' (for saving figures) and 'Res' (for saving data results) need to be constructed first. 

To be able to run the notebooks themselves, it is most convenient to use the supplied conda environment file ([environment.yml](environment.yml)) to create the corresponding conda environment as described in Section 'Tutorial' or [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Data can be downloaded from scratch using the script 'Omni_dataset_generator.py' or the data saved folder 'Data'. 'main.py' is an End-to-End pipeline of, 1) using the downloaded data and GRU method to develope a Dst model; 2) using ACCRUE method and the developed Dst model to develop a dDst (i.e., uncertainty of Dst) model; and 3) assimilate dDst model into Dst model to further improve the performance of the model. 

### Python modules

The following files contain useful functions and classes used throughout the notebooks.

- [End-to-end pipeline](Main_boost_multi.py) : an end-to-end pipeline from ACE data and Dst to model Dst & dDst, and eventually assimilate dDst into Dst model. This model achieves state of the art results in multiple hour ahead forecasting of Dst, which has been a primary focus of the community.
- [Functions](funs.py) : Various functions ranging from model training to plotting.
- [networks](nets.py) : arritectures used for training.
- [real-data download](ACE_dataset_generator.py) : Download the original measurements from ACE or DSCOVER in the same format with Data/Omni_data.pkl.

### Data

Data are organized by files containing different datasets organized roughly by source.

- [Original data set](Data/Omni_data.pkl) : All measurements included in the OMNI low-res dataset without flags and uncertainties, together with Dst from Kyoto. Gaps of 72 hours or less have been filled via linear interpolation. This dataset contains all OMNI low res data with no-data values replaced by np.nans.

- [ML-ready data set](Data/data_1_-100.h5) : The name of data is 'Data/data_(delay)_(Dst_sel).h5'. 

Inside file, 
'X_DL_(idx_storm)': independent variable set for storm (idx_storm) whose size is (number of samples, width of window, number of variables), where width of window is normally set to 6 (t-6:t). All 8 variables are:

0: IMF electron density(N)
1: IMF electron velocity(V)
2: IMF B_norm(i.e., sqrt(Bx^2+By^2+Bz^2))
3: IMF Bz
4: F10.7 one day ago
5: dipole tilt angle:
t_year = 23.4*np.cos((DOY-172)*2*pi/365.25)
t_day = 11.2*np.cos((UTC-16.72)*2*pi/24)
t = t_year+t_day
6: sin of clock angle: sin(arctan(IMF By, IMF Bz))
7: Dst

'Y_DL_(idx_storm)': target (Dst(t+delay)) set for storm (idx_storm) whose size is (number of samples, width of window, 1). 

'Dst_Per(idx_storm)': target (Dst(t)) predictions from persistence model for storm (idx_storm) whose size is (number of samples, ). Only valid for delay=1 if iteration mode is on.

- [Dst & dDst predictions](Data/Uncertainty_1_-100.h5) : The name of data is 'Data/Uncertainty_(delay)_(Dst_sel).h5'. 
Inside file, 
'y(idx_storm)': Dst model prediction for training set;
'y_t(idx_storm)': Dst model prediction for test set;
'std(idx_storm)': dDst model prediction for training set;
'std_t(idx_storm)': dDst model prediction for test set;

Their size is always (number of samples, ).

- [Dst model's coefficients](Data/params_new_1_-100-32-0resi_std.pt) : The name of data is 'Data/params_new_(delay)--(Dst_sel)-(idx_storm)-(boost_number)(criteria).pt'. 

- [dDst model's coefficients](Data/params_std_1_-100-32-1gruresi_std.pt) : The name of data is 'Data/params_new_(delay)_(Dst_sel)-(idx_storm)-(boost_num)(target)(criteria).pt'. Where the 'target' is either 'gru' or 'per' means either dDst of either GRU model or persistence model predictions; 'boost_num' is between 0-10 here (max can be changed); criteria is 'resi_std'.


# Tutorial

## create folder for generated images and results

    mkdir Figs Res

## environment install 

    conda env create -f environment.yml
    
Then activate LiveDst by

    conda activate LiveDst

## Omni data download

    python3 Omni_dataset_generator.py -t0 2000 1 1 -t1 2022 1 1 -filename Data/Omni_data.pkl

The data has been downloaded and saved in Data/Omni_data.pkl (in case no Internet)

## end-to-end python scripts

    python3 main_boost_multi.py -boost_num 10 -DA_num 10 -boost_method linear -criteria resi_std -model GRU -Omni_data Data/all_19990101-20170501.pkl -ratio 1.0 -train_per 5.0 -smooth_width 3 -DA_method Linear -Dst_sel -100 -std_method GRU -device 3 -storm_idx 32 -iter_flag -pred_plot -removal -final_plot -var_idx 0 1 2 3 4 5 6 -delay 6

set -device >=10 to use cpu.