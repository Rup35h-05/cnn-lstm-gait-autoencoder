import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from gait_autoencoder import CNNLSTMAutoencoder

# =========================
# LOAD VIDEO FRAMES
# =========================


def load_frames(video_path, num_frames=16, img_size=112):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Could not read video")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.array(frames).astype("float32") / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))
    return torch.from_numpy(frames)


# =========================
# MAIN
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = CNNLSTMAutoencoder()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    model.to(device)
    model.eval()

    # Load video
    frames = load_frames(args.video).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        recon = model(frames)

    # Compute MSE per frame
    mse_list = []
    for t in range(frames.shape[1]):
        mse = nn.functional.mse_loss(
            recon[0, t],
            frames[0, t]
        ).item()
        mse_list.append(mse)

    # Save anomaly scores
    df = pd.DataFrame({
        "frame": list(range(len(mse_list))),
        "mse": mse_list
    })
    df.to_csv(
        os.path.join(args.out_dir, "anomaly_scores.csv"),
        index=False
    )

    # Plot anomaly curve
    plt.figure()
    plt.plot(mse_list)
    plt.xlabel("Frame")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("Anomaly Curve")
    plt.savefig(
        os.path.join(args.out_dir, "anomaly_curve.png")
    )
    plt.close()

    # Generate heatmaps (first 3 frames)
    for i in range(min(3, frames.shape[1])):
        diff = torch.abs(
            recon[0, i] - frames[0, i]
        ).mean(dim=0).cpu().numpy()

        plt.figure()
        plt.imshow(diff, cmap="hot")
        plt.axis("off")
        plt.title(f"Frame {i} Heatmap")
        plt.savefig(
            os.path.join(
                args.out_dir,
                f"frame_{i:02d}_heatmap.png"
            )
        )
        plt.close()

    print("Visualization complete")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
