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
complex_masks_method = 0


#-------------------------------------------------

# this code processes all the data we need from ./LibriSpeech

#-------------------------------------------------

# create /data repertory for all data files we will create
if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/"+"data") :
    os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data")

# Raw dataset Path
#source_audio_dir = "../LibriSpeech/dev-clean"
source_audio_dir = "/content/drive/MyDrive/DL_Speech_Denoising/LibriSpeech_7G/train-clean-100"


#--------------------------------------------------------------------------

# create clean dataset from raw LibriSpeech dataset

if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset") :
    os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset")

clean_data_path = "/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset"

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
if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset_norm") :
    os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset_norm")

norm_clean_data_path = "/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset_norm"

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
    cafet_noise, sr = librosa.load("/content/drive/MyDrive/DL_Speech_Denoising/babble_16k.wav", sr = sr)
    len_noise = len(cafet_noise)
    t = randint(0, len_noise - duree*sr)
    cafet_noise = cafet_noise[t : t+duree*sr]
    return cafet_noise 

def normalize_noise_to_clean(clean, noise):  # ensures noise and speech have the same power
    rms_clean = np.sqrt(np.mean(clean**2)) # mean power of speech
    rms_noise = np.sqrt(np.mean(noise**2)) # mean power of noise
    noise = noise * (rms_clean / rms_noise) # normalisation wrt speech power
    return noise

# creating the noisy dataset : clean dataset + noise generation

# alpha = 0.5 ## alpha = 0.5 ->  SNR IN = 6dB 
alpha = 1 ## alpha = 1 ->  SNR IN = 0dB 

if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/noisy_dataset") :
    os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/noisy_dataset")

if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/noise_only_dataset") :
    os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/noise_only_dataset")


noisy_data_path = "/content/drive/MyDrive/DL_Speech_Denoising/data/noisy_dataset"
noise_only_path = "/content/drive/MyDrive/DL_Speech_Denoising/data/noise_only_dataset"

norm_clean_files = sorted(os.listdir(norm_clean_data_path)) 

if len(os.listdir(noisy_data_path)) == 0:
    for filename in tqdm(norm_clean_files, total=len(norm_clean_files), desc="Creating noisy datasets"):
        source_path = norm_clean_data_path + "/" +  filename
        dest_path = noisy_data_path + "/"+ "noised_" + filename
        dest_noise_path = noise_only_path + "/" + "noise_" + filename
        y, sr = librosa.load(source_path, sr=sr)
        noise = rand_noise_generation(duree = audio_duration)
        noise = normalize_noise_to_clean(y, noise)
        y_noisy = y + alpha * noise

        y = y + alpha * noise
        sf.write(dest_path, y, sr)
        sf.write(dest_noise_path, noise, sr)

#--------------------------------------------------------------------------

# FIRST APPROCH  : BINARY MASKS 

win_length = 400
n_fft = 510
hop_length = 188
window = 'hann'


if binary_method:

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_speech") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_speech")

    preprocessed_mask_pure_speech_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_speech"


    if len(os.listdir(preprocessed_mask_pure_speech_dir)) == 0:
        for filename in tqdm(norm_clean_files, total=len(norm_clean_files), desc="Creating speech pure spectrograms"):
            source_path = os.path.join(norm_clean_data_path, filename)
            filename = filename.replace('.wav', '') # pour enlever le .wav à la fin du nom du fichier et le remplacer par ''
            dest_path = os.path.join(preprocessed_mask_pure_speech_dir, filename)
            y, sr = librosa.load(source_path, sr = sr)
            spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            spectrogramme_mod_squarred = librosa.util.abs2(spectrogramme)
            np.save(dest_path + '.npy', spectrogramme_mod_squarred)


    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_speech_plus_noise") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_speech_plus_noise")

    speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/noisy_dataset"
    preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_speech_plus_noise"

    speech_plus_noise_files = sorted(os.listdir(speech_plus_noise_dir))

    if len(os.listdir(preprocessed_mask_speech_plus_noise_dir)) == 0:
        for filename in tqdm(speech_plus_noise_files, total=len(speech_plus_noise_files), desc="Creating noise+speech spectrograms"):
            source_path = os.path.join(speech_plus_noise_dir, filename)
            filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
            dest_path = os.path.join(preprocessed_mask_speech_plus_noise_dir, filename)
            y, sr = librosa.load(source_path, sr = sr)
            spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            spectrogramme_mod_squarred = librosa.util.abs2(spectrogramme)
            np.save(dest_path + '.npy', spectrogramme_mod_squarred)


    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_noise") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_noise")

    noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/noise_only_dataset"
    preprocessed_mask_pure_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_noise"

    noise_files = sorted(os.listdir(noise_dir))

    if len(os.listdir(preprocessed_mask_pure_noise_dir)) == 0:
        for filename in tqdm(noise_files, total=len(noise_files), desc="Creating noise pure spectrograms"):
            source_path = os.path.join(noise_dir, filename)
            filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
            dest_path = os.path.join(preprocessed_mask_pure_noise_dir, filename)
            y, sr = librosa.load(source_path, sr = sr)
            spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            spectrogramme_mod_squarred = librosa.util.abs2(spectrogramme)
            np.save(dest_path + '.npy', spectrogramme_mod_squarred)
                

    # creating target masks for further DL-model training 

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/mask_target") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/mask_target")

    mask_target = "/content/drive/MyDrive/DL_Speech_Denoising/data/mask_target"
    preprocessed_mask_pure_speech_files = sorted(os.listdir(preprocessed_mask_pure_speech_dir))
    
    if len(os.listdir(mask_target)) == 0:
            for filename in tqdm(preprocessed_mask_pure_speech_files, total=len(preprocessed_mask_pure_speech_files), desc="Creating masks"):
                
                path_noise = os.path.join(preprocessed_mask_pure_noise_dir, "noise_"+filename)
                path_speech = os.path.join(preprocessed_mask_pure_speech_dir, filename)
                dest_path = os.path.join(mask_target, "mask_"+filename)
                noise = np.load(path_noise)
                speech = np.load(path_speech)
                diff = np.subtract(speech, noise)

                mask = np.zeros([diff.shape[0], diff.shape[1]])
                mask = (diff > 0).astype(int)
                np.save(dest_path, mask)
    

if complex_masks_method:

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_speech") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_speech")

    Re_preprocessed_mask_pure_speech_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_speech"

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_speech") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_speech")

    Im_preprocessed_mask_pure_speech_dir = "../data/complex/Im_preprocessed_mask_pure_speech"


    if len(os.listdir(Re_preprocessed_mask_pure_speech_dir)) == 0:
        for filename in tqdm(norm_clean_files, total=len(norm_clean_files), desc="Creating speech pure spectrograms"):
            source_path = os.path.join(norm_clean_data_path, filename)
            filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
            Re_dest_path = os.path.join(Re_preprocessed_mask_pure_speech_dir, filename)
            Im_dest_path = os.path.join(Im_preprocessed_mask_pure_speech_dir, filename)
            y, sr = librosa.load(source_path, sr = sr)
            spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            Re = np.real(spectrogramme)
            Im = np.imag(spectrogramme)
            np.save(Re_dest_path + '.npy', Re)
            np.save(Im_dest_path + '.npy', Im)


    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_speech_plus_noise") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_speech_plus_noise")

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_speech_plus_noise") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_speech_plus_noise")

    speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/noisy_dataset"
    Re_preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_speech_plus_noise"
    Im_preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_speech_plus_noise"

    speech_plus_noise_files = sorted(os.listdir(speech_plus_noise_dir))

    if len(os.listdir(Re_preprocessed_mask_speech_plus_noise_dir)) == 0:
        for filename in tqdm(speech_plus_noise_files, total=len(speech_plus_noise_files), desc="Creating noise+speech spectrograms"):
            source_path = os.path.join(speech_plus_noise_dir, filename)
            filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
            Re_dest_path = os.path.join(Re_preprocessed_mask_speech_plus_noise_dir, filename)
            Im_dest_path = os.path.join(Im_preprocessed_mask_speech_plus_noise_dir, filename)
            y, sr = librosa.load(source_path, sr = sr)
            spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            Re = np.real(spectrogramme)
            Im = np.imag(spectrogramme)
            np.save(Re_dest_path + '.npy', Re)
            np.save(Im_dest_path + '.npy', Im)


    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_noise") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_noise")

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_noise") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_noise")

    noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/noise_only_dataset"
    Re_preprocessed_mask_pure_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_noise"
    Im_preprocessed_mask_pure_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_noise"

    noise_files = sorted(os.listdir(noise_dir))

    if len(os.listdir(Re_preprocessed_mask_pure_noise_dir)) == 0:
        for filename in tqdm(noise_files, total=len(noise_files), desc="Creating noise pure spectrograms"):
            source_path = os.path.join(noise_dir, filename)
            filename = filename.replace('.wav', '') # pour enlever le .flac à la fin du nom du fichier
            Re_dest_path = os.path.join(Re_preprocessed_mask_pure_noise_dir, filename)
            Im_dest_path = os.path.join(Im_preprocessed_mask_pure_noise_dir, filename)
            y, sr = librosa.load(source_path, sr = sr)
            spectrogramme = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
            Re = np.real(spectrogramme)
            Im = np.imag(spectrogramme)
            np.save(Re_dest_path + '.npy', Re)
            np.save(Im_dest_path + '.npy', Im)
                

    # creating target masks for further DL-model training 

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_mask_target") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_mask_target")

    if not os.path.exists("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_mask_target") :
        os.mkdir("/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_mask_target")

    Re_mask_target = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_mask_target"
    Im_mask_target = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_mask_target"

    Re_preprocessed_mask_pure_speech_files = sorted(os.listdir(Re_preprocessed_mask_pure_speech_dir))
    Im_preprocessed_mask_pure_speech_files = sorted(os.listdir(Im_preprocessed_mask_pure_speech_dir))
    
    if len(os.listdir(Re_mask_target)) == 0:
            for filename in tqdm(Re_preprocessed_mask_pure_speech_files, total=len(Re_preprocessed_mask_pure_speech_files), desc="Creating masks"):
                
                Re_path_noise = os.path.join(Re_preprocessed_mask_pure_noise_dir, "noise_"+filename)
                Im_path_noise = os.path.join(Im_preprocessed_mask_pure_noise_dir, "noise_"+filename)
                Re_path_speech = os.path.join(Re_preprocessed_mask_pure_speech_dir, filename)
                Im_path_speech = os.path.join(Im_preprocessed_mask_pure_speech_dir, filename)
                Re_dest_path = os.path.join(Re_mask_target, "mask_"+filename)
                Im_dest_path = os.path.join(Im_mask_target, "mask_"+filename)
                Re_noise = np.load(Re_path_noise)
                Im_noise = np.load(Im_path_noise)
                Re_speech = np.load(Re_path_speech)
                Im_speech = np.load(Im_path_speech)
                Re_diff = np.subtract(Re_speech, Re_noise)
                Im_diff = np.subtract(Im_speech, Im_noise)

                Re_mask = np.zeros([Re_diff.shape[0], Re_diff.shape[1]])
                Im_mask = np.zeros([Im_diff.shape[0], Im_diff.shape[1]])
                Re_mask = (Re_diff > 0).astype(int)
                Im_mask = (Im_diff > 0).astype(int)
                np.save(Re_dest_path, Re_mask)
                np.save(Im_dest_path, Im_mask)
 
