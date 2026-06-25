import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# DATASET
# =========================
class GaitVideoDataset(Dataset):
    def __init__(self, manifest_csv, num_frames=16, img_size=112):
        self.df = pd.read_csv(manifest_csv)
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Cannot read video: {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return frames

    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]["video_path"]
        video = self.read_video(video_path)
        return video


# =========================
# CNN ENCODER
# =========================
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 14 * 14, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =========================
# CNN DECODER
# =========================
class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 14 * 14)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 14, 14)
        return self.deconv(x)


# =========================
# CNN + LSTM AUTOENCODER
# =========================
class CNNLSTMAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder = CNNDecoder(hidden_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        z = self.encoder(x)
        z = z.view(B, T, -1)

        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out.contiguous().view(B * T, -1)

        recon = self.decoder(lstm_out)
        recon = recon.view(B, T, C, H, W)
        return recon


# =========================
# TRAINING
# =========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = GaitVideoDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CNNLSTMAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    os.makedirs("../models", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for videos in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            videos = videos.to(device)
            optimizer.zero_grad()
            recon = model(videos)
            loss = criterion(recon, videos)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "../models/best.pth")

    print("Training complete. Best loss:", best_loss)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train(args)