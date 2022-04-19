import requests
import urllib
from os.path import exists

from ipdb import set_trace as st
from tqdm import tqdm

from datetime import datetime, timedelta
import datetime as dt

import numpy as np
import pandas as pd

from funs import MyBar

import matplotlib.pyplot as plt

import argparse

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-t0', type=int, nargs='+',
               default=[2000, 1, 1], help='Start date')
p.add_argument('-t1', type=int, nargs='+',
               default=[2021, 11, 10], help='Start date')
p.add_argument('-filename', type=str, default='Data/omni_data.pkl',
               help='Output file')
args = p.parse_args()

########################## config #########################
bar = MyBar()
start_date = args.t0
end_date = args.t1
# B_clu = ['ACE', 'DSCOVER'] # use DSCOVER as default
# SW_clu = ['ACE', 'ACE_real']

# B_vari = ['BGSE'] # or 'BGSM'
# SW_vari = ['Vp', 'Np']

########################## filename #########################

# Set the start and end date as year, month, day
t0 = datetime(*start_date).strftime("%Y-%m-%dT%H:%M:%S")
t1 = datetime(*end_date).strftime("%Y-%m-%dT%H:%M:%S")

t0_name = datetime(*start_date).strftime("%Y%m%d")
t1_name = datetime(*end_date).strftime("%Y%m%d")
# st()
url_head = "https://lasp.colorado.edu/space-weather-portal/latis/dap/"
url_date = "&time>="+t0+"Z&time<="+t1
url_end = "Z&formatTime(yyyy-MM-dd'T'HH:mm:ss)"

url_dst = "kyoto_dst_index.csv?time,dst"
url_ACE_B = "ac_h0_mfi.csv?time,BGSM._1,BGSM._2,BGSM._3"
url_ACE_B_realtime = "swpc_solar_wind_mag.csv?time,bx_gsm,by_gsm,bz_gsm"
url_DSCOVER_B = "dscovr_h0_mag.csv?time,B1GSE._1,B1GSE._2,B1GSE._3"
url_ACE_SW = "ac_h0_swe.csv?time,Np,Vp"
# url_ACE_SW_realtime = "ac_k0_swe.csv?time,Vp,Np"
url_ACE_SW_realtime = "swpc_solar_wind_plasma.csv?time,density,speed"
url_F107 = "penticton_radio_flux.csv?time,observed_flux"

# url_DSCOVER_B: after 2015.6.8 (larger), url_ACE_B: after 2017.7(smaller);
# url_ACE_SW: before 2017.7, url_ACE_SW_realtime: after 2017.7
if datetime(*start_date) < datetime.now() - timedelta(days=10):
    url_clu = [url_dst, url_F107, url_ACE_B, url_ACE_SW] #  
else:
    url_clu = [url_dst, url_F107, url_ACE_B_realtime, url_ACE_SW_realtime] #  
name = ['Dst', 'F107', 'B', 'SW'] # 

############### real measurements
for idx, url_t in tqdm(enumerate(url_clu)):
    url_name = url_head+url_t+url_date+url_end
    filename = 'Data/'+name[idx]+t0_name+'-'+t1_name+".csv"
    if exists(filename):
        pass
    else:
        urllib.request.urlretrieve(url_name, filename, \
            reporthook=bar.on_urlretrieve)

############### data preprocess

save_name = 'Data/all_'+t0_name+'-'+t1_name+'.pkl'
df_clu = []
for idx, name_t in tqdm(enumerate(name)):

    filename = 'Data/'+name_t+t0_name+'-'+t1_name+".csv"
    df_t = pd.read_csv(filename)[:-1]

    # st()
    # remove nan
    df_t.replace(-1e31, np.nan, inplace=True)
    time = \
        pd.to_datetime(df_t["time (yyyy-MM-dd'T'HH:mm:ss)"],
        infer_datetime_format=True,
        ).values
    del df_t["time (yyyy-MM-dd'T'HH:mm:ss)"]
    df_temp = pd.DataFrame(np.array(df_t), 
                           index=time,
                           columns=df_t.keys().values)
    
    # remove anomalies
    if name_t == 'SW':
        # st()
        df_temp[df_temp < 0] = np.nan
        df_temp = df_temp.interpolate(method='linear').ffill().bfill()
        print(df_temp.head())
    # intepolate the data to 1h
    df_temp = df_temp.resample('1h').mean()
    df_temp = df_temp.interpolate(method='linear').ffill().bfill()
    print(f'Missing value count {df_temp.isna().sum()}/{len(df_temp)}')

    df_clu.append(df_temp)

df = pd.concat(df_clu, axis=1)
# st()
if datetime(*start_date) < datetime.now() - timedelta(days=10):

    df = df.rename(columns={
                    'BGSM._1 (nT)': 'BX_GSM', 
                    'BGSM._2 (nT)': 'BY_GSM',
                    'BGSM._3 (nT)': 'BZ_GSM',
                    'dst (nT)': 'DST',
                    'observed_flux (solar flux unit (SFU))': 'F10_INDEX',
                    'Np (#/cc)': 'N',
                    'Vp (km/s)': 'V',
                    })
else:
    df = df.rename(columns={
                    'bx_gsm (nT)': 'BX_GSM', 
                    'by_gsm (nT)': 'BY_GSM',
                    'bz_gsm (nT)': 'BZ_GSM',
                    'dst (nT)': 'DST',
                    'observed_flux (solar flux unit (SFU))': 'F10_INDEX',
                    'density (cm^-3)': 'N',
                    'speed (km/s)': 'V',
    })
# st()
df.to_pickle(save_name)

# fig, ax = plt.subplots(figsize=(16, 16))
# df.plot(subplots=True, ax=ax)
