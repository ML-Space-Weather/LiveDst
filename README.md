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

    python3 main.py -delay 1 -storm_idx 27 -model GRU -pred_flag -ratio 1.1 -smooth_width 3 -iter_flag -pred_plot -std_method MLP -device 7 -QQplot -Dst_flag -std_flag -DA_method Linear

set -device >=10 to use cpu.

The results figure can be found as 'Figs/predict_UQ2_'+delay+'--'+Dst_sel+'-'+storm_index+'.jpg'