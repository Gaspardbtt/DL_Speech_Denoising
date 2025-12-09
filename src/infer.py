import torch
import torch.nn as nn
from model import *
import os 
from os import listdir, mkdir, system
import librosa
from os.path import isdir
import numpy as np
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler

binary_method = 0
complex_masks_method = 0
temporal_method = 1

if binary_method :

    def normalize_noisy_to_clean(clean, noisy):  # ensures noisy output and clean speech have the same power
        rms_clean = np.sqrt(np.mean(clean**2)) # mean power of speech
        rms_noisy = np.sqrt(np.mean(noisy**2)) # mean power of noisy
        noisy = noisy * (rms_clean / rms_noisy) # normalisation wrt speech power
        return noisy

    # test files source 
    test_loader = torch.load("../data/test_loader/test_loader.pt", weights_only=False)
    X_test_names = np.load("../data/named/names.npy")

    # STFT params 
    win_length = 400
    n_fft = 510
    hop_length = 188
    window = 'hann'
    sr = 16000

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = UNet()
    model.to(device) 
    model.load_state_dict(torch.load("../models/Unet.pt"))
    model.eval()


    Spectrograms_squarred_batch = []
    with torch.no_grad():
        test_loss = 0.0
        for X_batch, y_batch in test_loader :
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            # outputs = (outputs > 0.5).float() # garder si on veut binary mask.
            spectrogram_squarred = outputs.mul(X_batch)
            spectrogram_squarred = spectrogram_squarred.detach().cpu()
            spectrogram_squarred = spectrogram_squarred.numpy()
            spectrogram_squarred_npy = spectrogram_squarred.squeeze(1)
            Spectrograms_squarred_batch.append(spectrogram_squarred_npy)
            

    def griffinlim_with_phase(mag, phase_init, n_iter, n_fft, hop_length, win_length, window):
        S = mag * np.exp(1j * phase_init)
        for _ in range(n_iter):
            x = librosa.istft(S, hop_length=hop_length, win_length=win_length, window=window)
            S_est = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
            S = mag * np.exp(1j * np.angle(S_est))
        return librosa.istft(S, hop_length=hop_length, win_length=win_length, window=window)



    if not isdir("../data/synth_data_soft"):
        os.mkdir("../data/synth_data_soft")
    output_dir = "../data/synth_data_soft"
    noisy_dataset_dir = "../data/noisy_dataset"
    clean_dataset_dir = "../data/clean_dataset_norm"

    # Normalisation norm_file{i}...
    noisy_dataset_files = [f.replace("noised_", "", 1).replace(".wav", "") for f in os.listdir(noisy_dataset_dir)]
    X_test_names = [f.replace(".npy", "") for f in X_test_names]
    

    index = 0
    n_iter = 120   # nombre d’itérations algorithme GL

    for batch_idx, spec_squarred_batch in enumerate(Spectrograms_squarred_batch):
        batch_len = spec_squarred_batch.shape[0]
        for j in range(batch_len):
            clean_name = X_test_names[index]   
            idx = noisy_dataset_files.index(clean_name)
            noisy_file_path = os.path.join(noisy_dataset_dir, f"noised_{clean_name}.wav")
            clean_file_path = os.path.join(clean_dataset_dir, f"{clean_name}.wav")
            y_noisy, rate = librosa.load(noisy_file_path, sr=sr)
            Y_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
            phase_noisy = np.angle(Y_noisy)
            spectrogram_squarred_mag = spec_squarred_batch[j]
            spectrogram_mag = np.sqrt(spectrogram_squarred_mag)
            y_synth = griffinlim_with_phase(
                mag=spectrogram_mag,
                phase_init=phase_noisy,
                n_iter=n_iter,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window
            )
            y_clean, _ = librosa.load(clean_file_path, sr = None)
            y_synth = normalize_noisy_to_clean(y_clean, y_synth)
            sf.write(f"{output_dir}/{clean_name}.wav", y_synth, sr)
            index += 1


if complex_masks_method :

    # test files source 
    Re_test_loader = torch.load("../data/test_loader/Re_test_loader.pt", weights_only=False)
    Im_test_loader = torch.load("../data/test_loader/Im_test_loader.pt", weights_only=False)
    X_test_names = np.load("../data/named/names.npy")

    # STFT params 
    win_length = 400
    n_fft = 510
    hop_length = 188
    window = 'hann'
    sr = 16000

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    Re_model = UNet()
    Re_model.to(device) 
    Re_model.load_state_dict(torch.load("../models/Re_Unet.pt"))
    Re_model.eval()

    Im_model = UNet()
    Im_model.to(device) 
    Im_model.load_state_dict(torch.load("../models/Im_Unet.pt"))
    Im_model.eval()


    Re_Spectrograms_batch = []
    Im_Spectrograms_batch = []
    with torch.no_grad():
        test_loss = 0.0
        for X_batch, y_batch in Re_test_loader :
            X_batch = X_batch.to(device)
            X_batch = X_batch / torch.max(X_batch)
            y_batch = y_batch.to(device)
            outputs = Re_model(X_batch)
            spectrogram = outputs.mul(X_batch)
            spectrogram = spectrogram.detach().cpu()
            spectrogram = spectrogram.numpy()
            spectrogram_npy = spectrogram.squeeze(1)
            Re_Spectrograms_batch.append(spectrogram_npy)

        for X_batch, y_batch in Im_test_loader :
            X_batch = X_batch.to(device)
            X_batch = X_batch / torch.max(X_batch)
            y_batch = y_batch.to(device)
            outputs = Im_model(X_batch)
            spectrogram = outputs.mul(X_batch)
            spectrogram = spectrogram.detach().cpu()
            spectrogram = spectrogram.numpy()
            spectrogram_npy = spectrogram.squeeze(1)
            Im_Spectrograms_batch.append(spectrogram_npy)


    if not isdir("../data/complex/synth_data"):
        os.mkdir("../data/complex/synth_data")
    output_dir = "../data/complex/synth_data"
    noisy_dataset_dir = "../data/noisy_dataset"

    # Normalisation norm_file{i}...
    noisy_dataset_files = [f.replace("noised_", "", 1).replace(".wav", "") for f in os.listdir(noisy_dataset_dir)]
    X_test_names = [f.replace(".npy", "") for f in X_test_names]


    index = 0

    for batch_idx, Re_spec_batch in enumerate(Re_Spectrograms_batch):

        batch_len = Re_spec_batch.shape[0]

        for j in range(batch_len):

            clean_name = X_test_names[index]   

            idx = noisy_dataset_files.index(clean_name)

            noisy_file_path = os.path.join(noisy_dataset_dir, f"noised_{clean_name}.wav")

            # y_noisy, rate = librosa.load(noisy_file_path, sr=sr)

            # Y_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
            # phase_noisy = np.angle(Y_noisy)

            Im_spec_batch = Im_Spectrograms_batch[batch_idx]

            spectro_complex = Re_spec_batch[j] + 1j * Im_spec_batch[j]

            y_synth = librosa.istft(
                spectro_complex,
                hop_length=hop_length,
                win_length=win_length,
                window=window
            )


            # spectrogram_mag = 10 * spec_batch[j]

            # y_synth = griffinlim_with_phase(
            #     mag=spectrogram_mag,
            #     phase_init=phase_noisy,
            #     n_iter=n_iter,
            #     n_fft=n_fft,
            #     hop_length=hop_length,
            #     win_length=win_length,
            #     window=window
            # )

            sf.write(f"{output_dir}/{clean_name}.wav", y_synth, sr)

            index += 1


if temporal_method:

    sr = 16000

    test_loader = torch.load("../data/test_loader/test_loader_temp.pt", weights_only=False)
    X_test_names = np.load("../data/named/names.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = DemucsLike()
    model.to(device)
    model.load_state_dict(torch.load("../models/DemucsLike.pt", map_location=device))
    model.eval()

    if not os.path.isdir("../data/synth_data_temp"):
        os.mkdir("../data/synth_data_temp")

    output_dir = "../data/synth_data_temp"
    clean_dataset_dir = "../data/clean_dataset_norm"

    X_test_names = [f.replace(".npy", "") for f in X_test_names]

    outputs_idx = 0

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch) 
            outputs = outputs.cpu().numpy()

            batch_size = outputs.shape[0]

            for j in range(batch_size):

                clean_name = X_test_names[outputs_idx]
                y_pred = outputs[j][0] 

                sf.write(f"{output_dir}/{clean_name}", y_pred, samplerate = sr)

                outputs_idx += 1
