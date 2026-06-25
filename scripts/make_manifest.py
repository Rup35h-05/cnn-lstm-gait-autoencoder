import os
import csv

# 🔴 IMPORTANT: CHANGE THIS if your videos are in a different folder
VIDEO_DIR = r"/Users/user/Downloads/CV_PROJECT/data/videos"
OUT_FILE  = r"/Users/user/Downloads/CV_PROJECT/data/manifests/dataset_manifest.csv"

os.makedirs(r"F:\CV_PROJECT\data\manifests", exist_ok=True)

rows = []
files = os.listdir(VIDEO_DIR)

print("✅ Files found in video directory:")
for f in files:
    print("  ", f)

for fname in sorted(files):
    if fname.lower().endswith(".mp4"):
        subject_id = fname.split("_")[0]   # PDFE01
        full_path = os.path.join(VIDEO_DIR, fname)
        rows.append([full_path, subject_id])

if len(rows) == 0:
    print(" ERROR: No .mp4 files found in", VIDEO_DIR)
else:
    print(f" Found {len(rows)} video files")

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "subject_id"])
    writer.writerows(rows)

print(" Manifest written to:", OUT_FILE)
