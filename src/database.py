import soundfile as sf
import librosa
import librosa.display
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from os import listdir, mkdir, system
from os.path import join, isdir, basename, splitext
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import glob
import shutil
import re
import pickle 

import matplotlib.pyplot as plt


#-------------------------------------------------

# this code processes all the data we need from ./LibriSpeech

#-------------------------------------------------

