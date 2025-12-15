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
import argparse
from numpy import linalg


from model import *


binary_method = 0
complex_masks_method = 0
temporal_method = 1


#------------------------------------------------------------------------------------------------------------

if complex_masks_method :

    # Re training
    Re_preprocessed_mask_speech_plus_noise_dir = "../data/complex/Re_preprocessed_mask_speech_plus_noise"
    Re_preprocessed_mask_pure_speech_noise_dir = "../data/complex/Re_preprocessed_mask_pure_speech"

    Im_preprocessed_mask_speech_plus_noise_dir = "../data/complex/Im_preprocessed_mask_speech_plus_noise"
    Im_preprocessed_mask_pure_speech_noise_dir = "../data/complex/Im_preprocessed_mask_pure_speech"

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

    os.makedirs("../data/test_loader", exist_ok=True)
    torch.save(test_loader, "../data/test_loader/Re_test_loader.pt")



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
                torch.save(model.state_dict(), "../models/Re_Unet.pt")
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

    Re_preprocessed_mask_speech_plus_noise_dir = "../data/complex/Re_preprocessed_mask_speech_plus_noise"
    Re_preprocessed_mask_pure_speech_noise_dir = "../data/complex/Re_preprocessed_mask_pure_speech"

    Im_preprocessed_mask_speech_plus_noise_dir = "../data/complex/Im_preprocessed_mask_speech_plus_noise"
    Im_preprocessed_mask_pure_speech_noise_dir = "../data/complex/Im_preprocessed_mask_pure_speech"

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

    os.makedirs("../data/test_loader", exist_ok=True)
    torch.save(test_loader, "../data/test_loader/Im_test_loader.pt")



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
                torch.save(model.state_dict(), "../models/Im_Unet.pt")
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
    preprocessed_mask_speech_plus_noise_dir = "../data/preprocessed_mask_speech_plus_noise"
    preprocessed_mask_pure_speech_noise_dir = "../data/preprocessed_mask_pure_speech"
    mask_target_dir = "../data/mask_target"
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

    os.makedirs("../data/test_loader", exist_ok=True)
    torch.save(test_loader, "../data/test_loader/test_loader.pt")



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
                torch.save(model.state_dict(), "../models/Unet.pt")
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

    # ======================================================
    # CUDA / A100 OPTIM
    # ======================================================
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ======================================================
    # ARGUMENTS TERMINAL
    # ======================================================
    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--clean_dir", type=str, default="../data/clean_dataset_norm")
        parser.add_argument("--noisy_dir", type=str, default="../data/noisy_dataset")
        parser.add_argument("--cache_path", type=str, default="../data/cache/dataset_temporal.pt")

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--epochs", type=int, default=15)
        parser.add_argument("--use_cache", action="store_true")

        parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])

        return parser.parse_args()


    # ======================================================
    # DEVICE
    # ======================================================
    def get_device(device_arg):
        if device_arg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_arg)


    # ======================================================
    # DATASET CREATION + CACHE
    # ======================================================
    def build_and_cache_dataset(clean_dir, noisy_dir, cache_path):
        X, y = [], []

        filenames = sorted(os.listdir(clean_dir))
        print(f"Building dataset ({len(filenames)} files)")

        for filename in tqdm(filenames, desc="Loading audio"):
            noisy_audio, _ = librosa.load(
                os.path.join(noisy_dir, "noised_" + filename), sr=None
            )
            clean_audio, _ = librosa.load(
                os.path.join(clean_dir, filename), sr=None
            )

            noisy = noisy_audio / (np.max(np.abs(noisy_audio)) + 1e-8)
            clean = clean_audio / (np.max(np.abs(clean_audio)) + 1e-8)

            noisy = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0)
            clean = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0)

            X.append(noisy)
            y.append(clean)

        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)

        split = int(0.8 * len(X))

        data = {
            "X_train": X[:split],
            "y_train": y[:split],
            "X_test": X[split:],
            "y_test": y[split:]
        }

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(data, cache_path)

        print(f"Dataset cached at {cache_path}")
        return data


    # ======================================================
    # LOSS
    # ======================================================
    def criterion(y_hat, y):
        resolutions = [(1024, 120, 600), (2048, 240, 1200)]

        # -------- Time-domain loss
        temp_loss = torch.norm(y - y_hat, p=1, dim=[1, 2]).mean()

        stft_losses = []
        for n_fft, hop, win in resolutions:
            window = torch.hann_window(win, device=y.device)

            Y = torch.stft(y.squeeze(1), n_fft, hop, win, window, return_complex=True)
            Y_hat = torch.stft(y_hat.squeeze(1), n_fft, hop, win, window, return_complex=True)

            Y_mag = torch.abs(Y)
            Y_hat_mag = torch.abs(Y_hat)

            mag_loss = torch.norm(
                torch.log10(Y_mag + 1e-8) - torch.log10(Y_hat_mag + 1e-8),
                p=1, dim=[1, 2]
            ).mean()

            sc_loss = (
                torch.norm(Y_mag - Y_hat_mag, p="fro", dim=[1, 2]) /
                (torch.norm(Y_mag, p="fro", dim=[1, 2]) + 1e-8)
            ).mean()

            phase_loss = torch.mean(
                Y_mag * (1.0 - torch.cos(torch.angle(Y) - torch.angle(Y_hat)))
            )

            stft_losses.append(mag_loss + sc_loss + 0.02 * phase_loss)

        return temp_loss + torch.stack(stft_losses).mean()


    # ======================================================
    # PARAM COUNT
    # ======================================================
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable


    # ======================================================
    # TRAINING
    # ======================================================
    def train(args):
        device = get_device(args.device)
        print(f"Device: {device}")

        if args.use_cache and os.path.exists(args.cache_path):
            print("Loading cached dataset")
            data = torch.load(args.cache_path, map_location="cpu")
        else:
            data = build_and_cache_dataset(
                args.clean_dir, args.noisy_dir, args.cache_path
            )

        train_loader = DataLoader(
            TensorDataset(data["X_train"], data["y_train"]),
            batch_size=args.batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(data["X_test"], data["y_test"]),
            batch_size=args.batch_size,
            shuffle=False
        )

        # ================= MODEL =================
        model = SimpleDemucs().to(device)

        total_params, trainable_params = count_parameters(model)
        print(f"Model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val = float("inf")
        train_losses, val_losses = [], []

        os.makedirs("../models", exist_ok=True)

        # ================= EPOCH LOOP =================
        for epoch in range(args.epochs):
            # -------- TRAIN --------
            model.train()
            total_loss = 0.0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]",
                leave=False
            )

            for X, y in train_pbar:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X.size(0)
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # -------- VALID --------
            model.eval()
            total_loss = 0.0

            val_pbar = tqdm(
                test_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [VAL]",
                leave=False
            )

            with torch.no_grad():
                for X, y in val_pbar:
                    X, y = X.to(device), y.to(device)
                    loss = criterion(model(X), y)
                    total_loss += loss.item() * X.size(0)
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_loss = total_loss / len(test_loader.dataset)
            val_losses.append(val_loss)

            # -------- SAVE --------
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "../models/DemucsLike.pt")

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            )

        # ================= PLOT =================
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.legend()
        plt.grid()
        plt.show()


    # ======================================================
    # MAIN
    # ======================================================
    if __name__ == "__main__":
        args = get_args()
        train(args)

            



    ##cmd :

    # python train_gpu.py \
    #     --epochs 30 \
    #     --batch_size 16 \
    #     --lr 0.0003 \
    #     --use_cache \
    #     --device cuda
