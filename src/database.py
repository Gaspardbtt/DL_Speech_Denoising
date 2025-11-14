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

# create /data repertory for all data files we will create
if not os.path.exists("../"+"data") :
    os.mkdir("../data")

# Raw dataset Path
source_audio_dir = "../LibriSpeech/dev-clean"

# create clean dataset from raw LibriSpeech dataset

if not os.path.exists("../data/clean_dataset") :
    os.mkdir("../data/clean_dataset")

clean_data_path = "../data/clean_dataset"


if len(os.listdir(clean_data_path)) == 0:
    
    i = 0 
    
    for path, dirs, files in os.walk(source_audio_dir):
        for filename in files:
            if filename.endswith(".flac"): 
                
                i += 1
                
                source_path = os.path.join(path, filename)
                new_filename = "file_" + f"{i}.wav"
                dest_path = os.path.join(clean_data_path, new_filename)

                # Copie + renomme en une fois
                shutil.copy(source_path, dest_path)



