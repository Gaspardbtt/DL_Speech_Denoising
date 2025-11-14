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
from random import *
from tqdm import tqdm 
import matplotlib.pyplot as plt

binary_method = 1   # 1 -> generate data to binary masks approch


#-------------------------------------------------

# this code processes all the data we need from ./LibriSpeech

#-------------------------------------------------

# create /data repertory for all data files we will create
if not os.path.exists("../"+"data") :
    os.mkdir("../data")

# Raw dataset Path
source_audio_dir = "../LibriSpeech/dev-clean"


#--------------------------------------------------------------------------

# create clean dataset from raw LibriSpeech dataset

if not os.path.exists("../data/clean_dataset") :
    os.mkdir("../data/clean_dataset")

clean_data_path = "../data/clean_dataset"

# count the number of files for tqdm display 
flac_files = []
for path, dirs, files in os.walk(source_audio_dir):
    for filename in files:
        if filename.endswith(".flac"):
            flac_files.append((path, filename))



if len(os.listdir(clean_data_path)) == 0:
    
    for i, (path, filename) in tqdm(enumerate(flac_files, start=1), total=len(flac_files), desc="Creating clean dataset"):
        
        source_path = os.path.join(path, filename)
        new_filename = f"file_{i}.wav"
        dest_path = os.path.join(clean_data_path, new_filename)

        shutil.copy(source_path, dest_path)


#--------------------------------------------------------------------------

# create a noisy dataset from the file babble_16k.wav

audio_duration = 3 # 3 secs 
sr = 16000

# creating 
if not os.path.exists("../data/clean_dataset_norm") :
    os.mkdir("../data/clean_dataset_norm")

norm_clean_data_path = "../data/clean_dataset_norm"

clean_files = sorted(os.listdir(clean_data_path))

if len(os.listdir(norm_clean_data_path)) == 0:
    for filename in tqdm(clean_files, total=len(clean_files), desc="Normalizing clean dataset"):

            y, sr = librosa.load(clean_data_path + "/" + filename, sr=sr)
            duration = librosa.get_duration(y = y, sr = sr)
            if duration > audio_duration :
                y = y[0:audio_duration * sr]
                sf.write(norm_clean_data_path + "/" + "norm_" + filename , y, sr)


# fonction that generate a random noise of 3 secs ( randomly choose that t_start of the noise in the file babble_16k.wav)

def rand_noise_generation(duree, sr = sr) :
    cafet_noise, sr = librosa.load("../babble_16k.wav", sr = sr)
    len_noise = len(cafet_noise)
    t = randint(0, len_noise - duree*sr)
    cafet_noise = cafet_noise[t : t+duree*sr]
    return cafet_noise 

# creating the noisy dataset : clean dataset + noise generation

alpha = 0.5 ## alpha = 0.5 ->  SNR IN = 6dB 

if not os.path.exists("../data/noisy_dataset") :
    os.mkdir("../data/noisy_dataset")

if not os.path.exists("../data/noise_only_dataset") :
    os.mkdir("../data/noise_only_dataset")


noisy_data_path = "../data/noisy_dataset"
noise_only_path = "../data/noise_only_dataset"

if len(os.listdir(noisy_data_path)) == 0:
    for path, dirs, files in os.walk(norm_clean_data_path):
        for filename in files:
            source_path = os.path.join(path, filename)    #os.path.join joint les deux parties du chemin path et filename pour avoir un chemin complet 
            dest_path = os.path.join(noisy_data_path, "noised_"+filename)   # idem pour la destination
            dest_noise_path = os.path.join(noise_only_path, "noise_"+ filename)
            y, sr = librosa.load(source_path, sr=sr)
            noise = rand_noise_generation(duree = audio_duration)
            y = y + alpha * noise
            sf.write(dest_path, y, sr)
            sf.write(dest_noise_path, noise, sr)

#--------------------------------------------------------------------------

# FIRST APPROCHE  : BINARY MASKS 

win_length = 400 
n_fft = 512
hop_length = win_length//2 
window = 'hann'



if binary_method:

    if not os.path.exists("../data/preprocessed_mask_pure_speech") :
        os.mkdir("../data/preprocessed_mask_pure_speech")

    preprocessed_mask_pure_speech_dir = "../data/preprocessed_mask_pure_speech"

    if len(os.listdir(preprocessed_mask_pure_speech_dir)) == 0:
        for path, dirs, files in os.walk(clean_data_path):
            for filename in files:
                source_path = os.path.join(path, filename)
                filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
                dest_path = os.path.join(preprocessed_mask_pure_speech_dir, filename)
                y, sr = librosa.load(source_path, sr = sr)
                spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
                spectrogramme_mod_squarred = librosa.util.abs2(spectrogramme)
                np.save(dest_path + '.npy', spectrogramme_mod_squarred)


    if not os.path.exists("../data/preprocessed_mask_speech_plus_noise") :
        os.mkdir("../data/preprocessed_mask_speech_plus_noise")

    speech_plus_noise_dir = "../data/noisy_dataset"
    preprocessed_mask_speech_plus_noise_dir = "../data/preprocessed_mask_speech_plus_noise"

    if len(os.listdir(preprocessed_mask_speech_plus_noise_dir)) == 0:
        for path, dirs, files in os.walk(speech_plus_noise_dir):
            for filename in files:
                source_path = os.path.join(path, filename)
                filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
                dest_path = os.path.join(preprocessed_mask_speech_plus_noise_dir, filename)
                y, sr = librosa.load(source_path, sr = sr)
                spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
                spectrogramme_mod_squarred = librosa.util.abs2(spectrogramme)
                np.save(dest_path + '.npy', spectrogramme_mod_squarred)


    if not os.path.exists("../data/preprocessed_mask_pure_noise") :
        os.mkdir("../data/preprocessed_mask_pure_noise")

    noise_dir = "../data/noise_only_dataset"
    preprocessed_mask_pure_noise_dir = "../data/preprocessed_mask_pure_noise"

    if len(os.listdir(preprocessed_mask_pure_noise_dir)) == 0:
        for path, dirs, files in os.walk(noise_dir):
            for filename in files:
                source_path = os.path.join(path, filename)
                filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
                dest_path = os.path.join(preprocessed_mask_pure_noise_dir, filename)
                y, sr = librosa.load(source_path, sr = sr)
                spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
                spectrogramme_mod_squarred = librosa.util.abs2(spectrogramme)
                np.save(dest_path + '.npy', spectrogramme_mod_squarred)
                

    
