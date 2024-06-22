# RAVEs
# Authors: Pan.LC
# Date: 2024/6/15

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.classification import MDM

from sklearn.ensemble import AdaBoostClassifier

from loaddata import Dataset_Left_Right_MI


dataset_name = 'Pan2023'

fs = 250
freqband = [1,100]
datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'

dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4,path=datapath)

