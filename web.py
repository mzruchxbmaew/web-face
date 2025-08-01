import os
import numpy as np
import streamlit as st # type: ignore

from deepface import DeepFace
from sklearn.cluster import DBSCAN # type: ignore
from sklearn.metrics.pairwise import cosine_distances # type: ignore
from PIL import Image

st.set_page_config(page_title="Face Grouping", layout="wide")
st.title("จัดกลุ่มภาพคนเดียวกัน (Facenet512 + DBSCAN)")

uploaded_files = st.file_uploader("อัปโหลดรูปหลายไฟล์", type=["jpg","jpeg","png"], accept_multiple_files=True)

distance_threshold = st.slider("ความเข้มงวด (eps)", 0.2, 0.6, 0.35, 0.01)

if st.button("ประมวลผล") and uploaded_files:
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        file_paths.append(path)

    st.write("เริ่มสร้าง embeddings ...")
    embeddings = []
    valid_files = []
    for path in file_paths:
        try:
            reps = DeepFace.represent(img_path=path, model_name="Facenet512", enforce_detection=True)
            embeddings.append(reps[0]["embedding"])
            valid_files.append(path)
            st.write(f"✓ {os.path.basename(path)}")
        except Exception as e:
            st.write(f"ข้าม {os.path.basename(path)}: {e}")

    embeddings = np.array(embeddings)
    distances = cosine_distances(embeddings)

    cluster = DBSCAN(eps=distance_threshold, min_samples=1, metric='precomputed')
    labels = cluster.fit_predict(distances)

    groups = {}
    for file, label in zip(valid_files, labels):
        groups.setdefault(label, []).append(file)

    st.subheader("ผลลัพธ์")
    for idx, files_in_group in groups.items():
        if len(files_in_group) < 2:
            continue
        st.markdown(f"### Group {idx}")
        cols = st.columns(5)
        col_idx = 0
        for file in files_in_group:
            img = Image.open(file)
            cols[col_idx].image(img, caption=os.path.basename(file), width=150)
            col_idx = (col_idx + 1) % 5