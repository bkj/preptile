#!/usr/bin/env python

"""
    plot.py
"""

import sys
import json
import pandas as pd

from rsub import *
from matplotlib import pyplot as plt

K = 25

df = pd.DataFrame([json.loads(x) for x in open(sys.argv[1])])

_ = plt.plot(df.train_acc.rolling(K).mean(), label='train')
_ = plt.plot(df.test_acc.rolling(K).mean(), label='test')
_ = plt.legend()
show_plot()