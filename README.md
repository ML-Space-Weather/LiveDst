# Tutorial

## Omni data download

python3 Omni_dataset_generator.py -t0 2000 1 1 -t1 2006 1 1 -filename Data/2000-2006.pkl

## end-to-end main scripts

python3 main.py -delay 3 -storm_idx 33 -model GRU -pred_flag -ratio 1.1 -smooth_width 3 -dst_flag -dst_flag -iter_flag -pred_plot -std_method GRU -device 7 -QQplot
