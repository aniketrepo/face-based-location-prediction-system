import cv2
import csv
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

from insightface.app import FaceAnalysis

# PATHS & CONFIG
BASE_DIR = Path(__file__).resolve().parent
MOBILITY_CSV = BASE_DIR / "data" / "mobility" / "identity_mobility.csv"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

SIMILARITY_THRESHOLD = 0.35   # lowered for single-embedding DB
LOCATION_SMOOTHING_WINDOW = 10

# LOAD MOBILITY DATA (CSV)
def load_identity_mobility(csv_path):
    mobility = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mobility[row["person_id"]].append({
                "place_name": row["place_name"],
                "place_type": row["place_type"],
                "days": row["days"].split("|"),
                "time_start": row["time_start"],
                "time_end": row["time_end"],
                "weight": float(row["weight"])
            })
    return mobility

# TIME UTILS
def time_in_range(now, start, end):
    now_t = now.time()
    start_t = datetime.strptime(start, "%H:%M").time()
    end_t = datetime.strptime(end, "%H:%M").time()

    if start_t <= end_t:
        return start_t <= now_t <= end_t
    else:
        return now_t >= start_t or now_t <= end_t

# LOCATION INFERENCE
def infer_location(identity, mobility_data, now):
    if identity not in mobility_data:
        return None

    today = now.strftime("%a")
    valid = []

    for loc in mobility_data[identity]:
        if today in loc["days"] and time_in_range(now, loc["time_start"], loc["time_end"]):
            valid.append(loc)

    if not valid:
        return None

    return max(valid, key=lambda x: x["weight"])

# COSINE SIMILARITY
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# LOAD EMBEDDINGS (ONE .NPY PER PERSON)
def load_known_embeddings(embeddings_dir):
    db = {}

    for f in embeddings_dir.glob("*.npy"):
        identity = f.stem
        emb = np.load(f)

        if emb.ndim != 1:
            raise ValueError(f"Embedding {f} is not 1D")

        db[identity] = emb

    print("Loaded identities:", list(db.keys()))
    return db

# MAIN
def main():
    # InsightFace init
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    known_db = load_known_embeddings(EMBEDDINGS_DIR)
    mobility_data = load_identity_mobility(MOBILITY_CSV)

    cap = cv2.VideoCapture(0)
    location_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        now = datetime.now()

        for face in faces:
            emb = face.normed_embedding

            identity = "Unknown"
            best_sim = 0.0

            for name, ref_emb in known_db.items():
                sim = cosine_similarity(emb, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    identity = name

            if best_sim < SIMILARITY_THRESHOLD:
                identity = "Unknown"

            location_text = "Unknown"

            if identity != "Unknown":
                loc = infer_location(identity, mobility_data, now)
                if loc:
                    location_buffer.append(loc["place_name"])

            if len(location_buffer) > LOCATION_SMOOTHING_WINDOW:
                location_buffer.pop(0)

            if location_buffer:
                location_text = Counter(location_buffer).most_common(1)[0][0]

            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"{identity} ({best_sim:.2f})",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"Likely at: {location_text}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        # UI hint
        cv2.putText(frame, "Press Q to quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Face-Based Location Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
