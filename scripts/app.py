import streamlit as st
import os
import tempfile
import subprocess
import pandas as pd

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Parkinson Gait Anomaly Detection",
    layout="wide"
)

st.title("🧠 Parkinson’s Gait Anomaly Detection")
st.write("CNN–LSTM Autoencoder based gait anomaly detection from video")

# -------------------------
# PATH SETUP (MAC SAFE)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best.pth")
VIS_SCRIPT = os.path.join(BASE_DIR, "visualize_anomaly.py")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "web_result")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_video = st.file_uploader(
    "Upload a gait video (.mp4)",
    type=["mp4"]
)

if uploaded_video is not None:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_video.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success("✅ Video uploaded successfully")
    st.video(video_path)

    # -------------------------
    # ANALYZE BUTTON
    # -------------------------
    if st.button("Analyze Gait"):
        st.info("Analyzing gait... please wait ⏳")

        cmd = [
            "python3",
            VIS_SCRIPT,
            "--video", video_path,
            "--checkpoint", MODEL_PATH,
            "--out_dir", OUTPUT_DIR
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            st.error("❌ Error during analysis")
            st.code(result.stderr)
        else:
            st.success("✅ Analysis completed successfully")

            # -------------------------
            # SHOW ANOMALY SCORE
            # -------------------------
            csv_path = os.path.join(OUTPUT_DIR, "anomaly_scores.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                mean_score = df["mse"].mean()
                st.metric(
                    "📌 Overall Anomaly Score (Mean MSE)",
                    f"{mean_score:.5f}"
                )

            # -------------------------
            # SHOW ANOMALY CURVE
            # -------------------------
            curve_path = os.path.join(OUTPUT_DIR, "anomaly_curve.png")
            if os.path.exists(curve_path):
                st.subheader("📈 Frame-wise Anomaly Curve")
                st.image(curve_path, use_column_width=True)

            # -------------------------
            # SHOW HEATMAPS
            # -------------------------
            st.subheader("🔥 Sample Heatmaps")
            cols = st.columns(3)
            for i in range(3):
                heatmap_path = os.path.join(
                    OUTPUT_DIR, f"frame_{i:02d}_heatmap.png"
                )
                if os.path.exists(heatmap_path):
                    cols[i].image(
                        heatmap_path,
                        caption=f"Frame {i}",
                        use_column_width=True
                    )
