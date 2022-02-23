#!/usr/bin/env python3

from datetime import datetime
from heliopy.data.omni import h0_mrg1hr
import argparse

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-t0', type=int, nargs='+',
               default=[2000, 1, 1], help='Start date')
p.add_argument('-t1', type=int, nargs='+',
               default=[2021, 11, 10], help='Start date')
p.add_argument('-filename', type=str, default='omni_data.pkl',
               help='Output file')
args = p.parse_args()

# Set the start and end date as year, month, day
t0 = datetime(*args.t0)
t1 = datetime(*args.t1)

# Download the data
ts = h0_mrg1hr(t0, t1)

# Extract dataframe
df = ts.to_dataframe()

# Save data so it can be loaded later
df.to_pickle(args.filename)
