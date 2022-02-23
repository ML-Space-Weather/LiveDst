# Tutorial

## create folder for generated images and results

    mkdir Figs Res

## environment install 

    conda env create -f environment.yml
    
Then activate LiveDst by
    conda activate LiveDst

## Omni data download

    python3 Omni_dataset_generator.py -t0 2000 1 1 -t1 2006 1 1 -filename Data/2000-2006.pkl

The data has been downloaded and saved in Data/omni_data.pkl (in case no Internet)

## end-to-end python scripts

    python3 main.py -delay 1 -storm_idx 33 -model GRU -pred_flag -ratio 1.1 -smooth_width 0 -dst_flag -dst_flag -iter_flag -pred_plot -std_method GRU -device 7 -QQplot
