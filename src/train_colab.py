from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir, mkdir, system
import os.path
import librosa
from numpy import linalg


from model import *


binary_method = 0
complex_masks_method = 0
temporal_method = 1


#------------------------------------------------------------------------------------------------------------

if complex_masks_method :

    # Re training
    Re_preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_speech_plus_noise"
    Re_preprocessed_mask_pure_speech_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_pure_speech"

    Im_preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_speech_plus_noise"
    Im_preprocessed_mask_pure_speech_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_speech"

    Re_mask_target_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_mask_target"
    Im_mask_target_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_mask_target"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    learning_rate = 0.0005
    num_epochs = 15


    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X, y = [], []

    name_files = sorted(os.listdir(Re_preprocessed_mask_pure_speech_noise_dir))

    for filename in tqdm(name_files,total=len(name_files),desc="Loading datasets for training"): 
        path_img_x = os.path.join(Re_preprocessed_mask_speech_plus_noise_dir, "noised_" + filename)
        path_img_y = os.path.join(Re_mask_target_dir, "mask_" + filename)
        scaler = MinMaxScaler()
        imgx = np.load(path_img_x)
        imgx = scaler.fit_transform(imgx)
        imgx = torch.from_numpy(imgx).float()
        imgy = np.load(path_img_y)
        imgy = torch.from_numpy(imgy).float()
        imgx = imgx.unsqueeze(1).permute(1, 0, 2) # pour avoir channels, height, width
        imgy = imgy.unsqueeze(1).permute(1, 0, 2)
        X.append([imgx, filename])
        y.append([imgy, filename])


    X_train_named = X[:int(len(X)*0.8)]
    X_test_named = X[int(len(X)*0.8):]

    y_train_named = y[:int(len(y)*0.8)]
    y_test_named = y[int(len(y)*0.8):]

    X_train = torch.stack([x[0] for x in X_train_named])
    y_train = torch.stack([x[0] for x in y_train_named])
    X_test = torch.stack([x[0] for x in X_test_named])
    y_test = torch.stack([x[0] for x in y_test_named])

    X_test_names = [x[1] for x in X_test_named]

    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/named", exist_ok=True)

    np.save("/content/drive/MyDrive/DL_Speech_Denoising/data/named/names" + '.npy', X_test_names)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  
    )

    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader", exist_ok=True)
    torch.save(test_loader, "/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader/Re_test_loader.pt")



    model = UNet().to(device)

    criterion = nn.BCELoss()  # loss binaire pcq mask binaire
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Counting params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}\n")
    print(f"device : {device}\n")
    # -------------------------------

    # training loop 

    print("--Neural Network training--")

    train_loss = []
    val_loss = []
    # model rep 
    os.makedirs("../models", exist_ok=True)


    pbar = tqdm(total=num_epochs, desc="epochs")


    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * X_batch.size(0)
        epoch_loss = training_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)



        # eval loop 

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                #outputs = (outputs >= 0.5).float()

                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            pred_bin = (outputs >= 0.5).float()
            accuracy = (pred_bin == y_batch).float().mean()
            val_loss.append(test_loss)
            m = np.min(val_loss)

            if test_loss <= m:
                # loading model
                torch.save(model.state_dict(), "/content/drive/MyDrive/DL_Speech_Denoising/models/Re_Unet.pt")
                opt_epoch = epoch
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Mean accuracy per pixels on last batch: {accuracy.item():.4f}")

        pbar.update(1)


    pbar.close()

    print(f"best model saved at epoch : {opt_epoch}/{num_epochs}")

    # plot training loss curve
    M1 = np.max(train_loss)
    M2 = np.max(val_loss)
    M = max(M1, M2)

    m1 = np.min(train_loss)
    m2 = np.min(val_loss)
    m = min(m1, m2)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation loss', color='orange')
    plt.xlabel('Epochs')
    plt.xlim(0,num_epochs)
    plt.ylim(m,M)
    plt.ylabel('Loss')
    plt.title('Train Loss / Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()


# Im training

    Re_preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Re_preprocessed_mask_speech_plus_noise"
    Re_preprocessed_mask_pure_speech_noise_dir = "../data/complex/Re_preprocessed_mask_pure_speech"

    Im_preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_speech_plus_noise"
    Im_preprocessed_mask_pure_speech_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/complex/Im_preprocessed_mask_pure_speech"

    Re_mask_target_dir = "../data/complex/Re_mask_target"
    Im_mask_target_dir = "../data/complex/Im_mask_target"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    learning_rate = 0.0005
    num_epochs = 15


    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X, y = [], []

    name_files = sorted(os.listdir(Im_preprocessed_mask_pure_speech_noise_dir))

    for filename in tqdm(name_files,total=len(name_files),desc="Loading datasets for training"): 
        path_img_x = os.path.join(Im_preprocessed_mask_speech_plus_noise_dir, "noised_" + filename)
        path_img_y = os.path.join(Im_mask_target_dir, "mask_" + filename)
        scaler = MinMaxScaler()
        imgx = np.load(path_img_x)
        imgx = scaler.fit_transform(imgx)
        imgx = torch.from_numpy(imgx).float()
        imgy = np.load(path_img_y)
        imgy = torch.from_numpy(imgy).float()
        imgx = imgx.unsqueeze(1).permute(1, 0, 2) # pour avoir channels, height, width
        imgy = imgy.unsqueeze(1).permute(1, 0, 2)
        X.append([imgx, filename])
        y.append([imgy, filename])


    X_train_named = X[:int(len(X)*0.8)]
    X_test_named = X[int(len(X)*0.8):]

    y_train_named = y[:int(len(y)*0.8)]
    y_test_named = y[int(len(y)*0.8):]

    X_train = torch.stack([x[0] for x in X_train_named])
    y_train = torch.stack([x[0] for x in y_train_named])
    X_test = torch.stack([x[0] for x in X_test_named])
    y_test = torch.stack([x[0] for x in y_test_named])

    X_test_names = [x[1] for x in X_test_named]

    os.makedirs("../data/named", exist_ok=True)

    np.save("../data/named/names" + '.npy', X_test_names)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  
    )

    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader", exist_ok=True)
    torch.save(test_loader, "/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader/Im_test_loader.pt")



    model = UNet().to(device)

    criterion = nn.BCELoss()  # loss binaire pcq mask binaire
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Counting params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}\n")
    print(f"device : {device}\n")
    # -------------------------------

    # training loop 

    print("--Neural Network training--")

    train_loss = []
    val_loss = []
    # model rep 
    os.makedirs("../models", exist_ok=True)


    pbar = tqdm(total=num_epochs, desc="epochs")


    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * X_batch.size(0)
        epoch_loss = training_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)



        # eval loop 

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                #outputs = (outputs >= 0.5).float()

                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            pred_bin = (outputs >= 0.5).float()
            accuracy = (pred_bin == y_batch).float().mean()
            val_loss.append(test_loss)
            m = np.min(val_loss)

            if test_loss <= m:
                # loading model
                torch.save(model.state_dict(), "/content/drive/MyDrive/DL_Speech_Denoising/models/Im_Unet.pt")
                opt_epoch = epoch
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Mean accuracy per pixels on last batch: {accuracy.item():.4f}")

        pbar.update(1)


    pbar.close()

    print(f"best model saved at epoch : {opt_epoch}/{num_epochs}")

    # plot training loss curve
    M1 = np.max(train_loss)
    M2 = np.max(val_loss)
    M = max(M1, M2)

    m1 = np.min(train_loss)
    m2 = np.min(val_loss)
    m = min(m1, m2)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation loss', color='orange')
    plt.xlabel('Epochs')
    plt.xlim(0,num_epochs)
    plt.ylim(m,M)
    plt.ylabel('Loss')
    plt.title('Train Loss / Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    #------------------------------------------------------------------------------------------------------------

if binary_method :

    # train loop for Binary masks approche
    preprocessed_mask_speech_plus_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_speech_plus_noise"
    preprocessed_mask_pure_speech_noise_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/preprocessed_mask_pure_speech"
    mask_target_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/mask_target"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    learning_rate = 0.0005
    num_epochs = 20


    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X, y = [], []

    name_files = sorted(os.listdir(preprocessed_mask_pure_speech_noise_dir))

    for filename in tqdm(name_files,total=len(name_files),desc="Loading datasets for training"): 
        path_img_x = os.path.join(preprocessed_mask_speech_plus_noise_dir, "noised_" + filename)
        path_img_y = os.path.join(mask_target_dir, "mask_" + filename)
        scaler = MinMaxScaler()
        imgx = np.load(path_img_x)
        imgx = scaler.fit_transform(imgx)
        imgx = torch.from_numpy(imgx).float()
        imgy = np.load(path_img_y)
        imgy = torch.from_numpy(imgy).float()
        imgx = imgx.unsqueeze(1).permute(1, 0, 2) # pour avoir channels, height, width
        imgy = imgy.unsqueeze(1).permute(1, 0, 2)
        X.append([imgx, filename])
        y.append([imgy, filename])


    X_train_named = X[:int(len(X)*0.8)]
    X_test_named = X[int(len(X)*0.8):]

    y_train_named = y[:int(len(y)*0.8)]
    y_test_named = y[int(len(y)*0.8):]

    X_train = torch.stack([x[0] for x in X_train_named])
    y_train = torch.stack([x[0] for x in y_train_named])
    X_test = torch.stack([x[0] for x in X_test_named])
    y_test = torch.stack([x[0] for x in y_test_named])

    X_test_names = [x[1] for x in X_test_named]

    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/named", exist_ok=True)

    np.save("/content/drive/MyDrive/DL_Speech_Denoising/data/named/names" + '.npy', X_test_names)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  
    )

    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader", exist_ok=True)
    torch.save(test_loader, "/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader/test_loader.pt")



    model = UNet().to(device)

    criterion = nn.BCELoss()  # loss binaire pcq mask binaire
    # criterion = nn.MSELoss() # si pas binaire
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Counting params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}\n")
    print(f"device : {device}\n")
    # -------------------------------

    # training loop 

    print("--Neural Network training--")

    train_loss = []
    val_loss = []
    # model rep 
    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/models", exist_ok=True)


    pbar = tqdm(total=num_epochs, desc="epochs")


    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * X_batch.size(0)
        epoch_loss = training_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)



        # eval loop 

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                #outputs = (outputs >= 0.5).float()

                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            pred_bin = (outputs >= 0.5).float()
            accuracy = (pred_bin == y_batch).float().mean()
            val_loss.append(test_loss)
            m = np.min(val_loss)

            if test_loss <= m:
                # loading model
                torch.save(model.state_dict(), "/content/drive/MyDrive/DL_Speech_Denoising/models/Unet.pt")
                opt_epoch = epoch
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Mean accuracy per pixels on last batch: {accuracy.item():.4f}")

        pbar.update(1)


    pbar.close()

    print(f"best model saved at epoch : {opt_epoch}/{num_epochs}")

    # plot training loss curve
    M1 = np.max(train_loss)
    M2 = np.max(val_loss)
    M = max(M1, M2)

    m1 = np.min(train_loss)
    m2 = np.min(val_loss)
    m = min(m1, m2)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation loss', color='orange')
    plt.xlabel('Epochs')
    plt.xlim(0,num_epochs)
    plt.ylim(m,M)
    plt.ylabel('Loss')
    plt.title('Train Loss / Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    #------------------------------------------------------------------------------------------------------------




if temporal_method :

    clean_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/clean_dataset_norm"
    noisy_dir = "/content/drive/MyDrive/DL_Speech_Denoising/data/noisy_dataset"

    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    learning_rate = 0.001
    num_epochs = 10

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X, y = [], []

    name_files = sorted(os.listdir(clean_dir))

    for filename in tqdm(name_files, total=len(name_files), desc="Loading datasets for training"):

        path_noisy = os.path.join(noisy_dir, "noised_" + filename)
        path_clean = os.path.join(clean_dir, filename)

        noisy_audio, sr = librosa.load(path_noisy, sr=None)  
        clean_audio, sr = librosa.load(path_clean, sr=None)

        scaler = MinMaxScaler()
        noisy_np = noisy_audio.reshape(-1, 1)       
        noisy_scaled = scaler.fit_transform(noisy_np).flatten()

        noisy = torch.from_numpy(noisy_scaled).float().unsqueeze(0).unsqueeze(0)
        clean = torch.from_numpy(clean_audio).float().unsqueeze(0).unsqueeze(0)  

        X.append([noisy, filename])
        y.append([clean, filename])


    X_train_named = X[:int(len(X)*0.8)]
    X_test_named = X[int(len(X)*0.8):]

    y_train_named = y[:int(len(y)*0.8)]
    y_test_named = y[int(len(y)*0.8):]

    X_train = torch.cat([x[0] for x in X_train_named], dim=0)
    y_train = torch.cat([x[0] for x in y_train_named], dim=0)
    X_test = torch.cat([x[0] for x in X_test_named], dim=0)
    y_test = torch.cat([x[0] for x in y_test_named], dim=0)

    X_test_names = [x[1] for x in X_test_named]
    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/named", exist_ok=True)
    np.save("/content/drive/MyDrive/DL_Speech_Denoising/data/named/names.npy", X_test_names)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader", exist_ok=True)
    torch.save(test_loader, "/content/drive/MyDrive/DL_Speech_Denoising/data/test_loader/test_loader_temp.pt")





    model = DemucsLike().to(device)

    # def criterion(y_hat_batch, y_batch, T) :
    #     resolutions = [[512, 50, 240], [1024, 120, 600], [2048, 240, 1200]] # [n_fft, hop_length, win_length]
    #     stft_losses = []
    #     for i in range(len(y_batch)) :
    #         y_hat = y_hat_batch[i]
    #         y = y_batch[i]            
    #         temp_loss = torch.norm(y - y_hat, p=1)
    #         for resolution in resolutions :
    #             Y_mag = np.abs(librosa.stft(y=y.detach().cpu().numpy(), n_fft=resolution[0], hop_length=resolution[1], win_length=resolution[2], window='hann'))
    #             Y_hat_mag = np.abs(librosa.stft(y=y_hat.detach().cpu().numpy(), n_fft=resolution[0], hop_length=resolution[1], win_length=resolution[2], window='hann'))
    #             mag_loss = (1/T)*torch.norm(np.log10(Y_mag)-np.log10(Y_hat_mag), p=1)
    #             sc_loss = torch.norm(Y_mag - Y_hat_mag, p='fro')/torch.norm(Y_mag, p='fro')
    #             stft_losses.append(mag_loss + sc_loss)
    #         loss = (1/T)*(np.sum(stft_losses) + temp_loss)
    #     return loss
    

    def criterion(y_hat_batch, y_batch, T):
        resolutions = [[512, 50, 240], [1024, 120, 600], [2048, 240, 1200]]  # [n_fft, hop_length, win_length]
        stft_losses = []

        for i in range(len(y_batch)):
            y_hat = y_hat_batch[i]
            y = y_batch[i]

            # L1 time-domain loss
            temp_loss = torch.norm(y - y_hat, p=1)

            for resolution in resolutions:
                n_fft, hop_length, win_length = resolution
                window = torch.hann_window(win_length, device=y.device)

                # STFT différentiable
                Y = torch.stft(y.squeeze(0), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
                Y_hat = torch.stft(y_hat.squeeze(0), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

                # Magnitude spectrogram
                Y_mag = torch.abs(Y)
                Y_hat_mag = torch.abs(Y_hat)

                # Log-magnitude L1 loss
                mag_loss = (1/T) * torch.norm(torch.log10(Y_mag + 1e-8) - torch.log10(Y_hat_mag + 1e-8), p=1)

                # Spectral convergence loss
                sc_loss = torch.norm(Y_mag - Y_hat_mag, p='fro') / (torch.norm(Y_mag, p='fro') + 1e-8)

                stft_losses.append(mag_loss + sc_loss)

            loss = (1/T) * (torch.sum(torch.stack(stft_losses)) + temp_loss)

        return loss




    # criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Counting params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}\n")
    print(f"device : {device}\n")
    # -------------------------------

    # training loop 

    print("--Neural Network training--")

    train_loss = []
    val_loss = []
    # model rep 
    os.makedirs("/content/drive/MyDrive/DL_Speech_Denoising/models", exist_ok=True)


    pbar = tqdm(total=num_epochs, desc="epochs")


    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch, T=len(y_batch[0]))
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * X_batch.size(0)
        epoch_loss = training_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)



        # eval loop 

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch, T=len(y_batch[0]))
                test_loss += loss.item() * X_batch.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            val_loss.append(test_loss)
            m = np.min(val_loss)

            if test_loss <= m:
                # loading model
                torch.save(model.state_dict(), "/content/drive/MyDrive/DL_Speech_Denoising/models/DemucsLike.pt")
                opt_epoch = epoch
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

        pbar.update(1)


    pbar.close()

    print(f"best model saved at epoch : {opt_epoch}/{num_epochs}")

    # plot training loss curve
    M1 = np.max(train_loss)
    M2 = np.max(val_loss)
    M = max(M1, M2)

    m1 = np.min(train_loss)
    m2 = np.min(val_loss)
    m = min(m1, m2)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation loss', color='orange')
    plt.xlabel('Epochs')
    plt.xlim(0,num_epochs)
    plt.ylim(m,M)
    plt.ylabel('Loss')
    plt.title('Train Loss / Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    #------------------------------------------------------------------------------------------------------------