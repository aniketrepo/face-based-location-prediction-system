import os
import cv2
import numpy as np
from face_utils import app

RAW_DIR = "data/raw_faces"
OUT_DIR = "data/embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

for person in os.listdir(RAW_DIR):
    person_path = os.path.join(RAW_DIR, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        embeddings.append(faces[0].embedding)

    if len(embeddings) == 0:
        print(f"[!] No face found for {person}")
        continue

    mean_embedding = np.mean(embeddings, axis=0)
    np.save(os.path.join(OUT_DIR, f"{person}.npy"), mean_embedding)

    print(f"[✓] Saved embedding for {person}")
