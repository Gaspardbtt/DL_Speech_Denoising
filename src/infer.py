import torch
import torch.nn as nn
from model import *
import os 
from os import listdir, mkdir, system
import librosa
from os.path import isdir
import numpy as np
import soundfile as sf


# test files source 
test_loader = torch.load("../data/test_loader/test_loader.pt")
X_test_names = np.load("../data/named/names.npy")

# STFT params 
win_length = 400
n_fft = 510
hop_length = 188
window = 'hann'
sr = 16000

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

model = CNN_MASK()
model.to(device) 
model.load_state_dict(torch.load("../models/model.pt"))
model.eval()


Spectrograms_batch = []
with torch.no_grad():
    test_loss = 0.0
    for X_batch, y_batch in test_loader :
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        outputs = (outputs > 0.5).float()
        spectrogram = outputs.mul(torch.sqrt(X_batch))
        spectrogram = spectrogram.detach().cpu()
        spectrogram = spectrogram.numpy()
        spectrogram_npy = spectrogram.squeeze(1)
        Spectrograms_batch.append(spectrogram_npy)
        

def griffinlim_with_phase(mag, phase_init, n_iter, n_fft, hop_length, win_length, window):
    S = mag * np.exp(1j * phase_init)
    for _ in range(n_iter):
        x = librosa.istft(S, hop_length=hop_length, win_length=win_length, window=window)
        S_est = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        S = mag * np.exp(1j * np.angle(S_est))
    return librosa.istft(S, hop_length=hop_length, win_length=win_length, window=window)



if not isdir("../data/synth_data"):
    os.mkdir("../data/synth_data")
output_dir = "../data/synth_data"
noisy_dataset_dir = "../data/noisy_dataset"

# Normalisation norm_file{i}...
noisy_dataset_files = [f.replace("noised_", "", 1).replace(".wav", "") for f in os.listdir(noisy_dataset_dir)]
X_test_names = [f.replace(".npy", "") for f in X_test_names]

index = 0
n_iter = 120   # nombre d’itérations algorithme

for batch_idx, spec_batch in enumerate(Spectrograms_batch):

    batch_len = spec_batch.shape[0]

    for j in range(batch_len):

        clean_name = X_test_names[index]   

        idx = noisy_dataset_files.index(clean_name)

        noisy_file_path = os.path.join(noisy_dataset_dir, f"noised_{clean_name}.wav")

        y_noisy, rate = librosa.load(noisy_file_path, sr=sr)

        Y_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        phase_noisy = np.angle(Y_noisy)

        spectrogram_mag = spec_batch[j]

        y_synth = griffinlim_with_phase(
            mag=spectrogram_mag,
            phase_init=phase_noisy,
            n_iter=n_iter,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )

        sf.write(f"{output_dir}/{clean_name}.wav", y_synth, sr)

        index += 1




