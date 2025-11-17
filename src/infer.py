import torch
import torch.nn as nn
from model import *
import os 
from os import listdir, mkdir, system
import librosa

# test files source 
test_loader = "blabla"  # liste des spectrogrames de test ( module carré )

# STFT params 
win_length = 400
n_fft = 510
hop_length = 188
window = 'hann'
sr = 16000

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


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
        



if not isdir("../data/synth_data"):
    os.mkdir("../data/synth_data")

# compteur pour aligner avec X_test_names
index = 0

for batch_idx, spec_batch in enumerate(Spectrograms_batch):

    batch_len = spec_batch.shape[0]

    for j in range(batch_len):

        spectrogram = spec_batch[j, :, :]

        y_synth = librosa.griffinlim(spectrogram, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)

        file_name = X_test_names[index]
        file_name = file_name.replace('.npy', '')

        sf.write(f"{output_dir}/synth_data/{file_name}.wav", y_synth, sr)

        index += 1
        