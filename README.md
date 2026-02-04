# Face-Based Location Prediction System
A real-time computer vision system that recognizes a person using face embeddings (ArcFace) and infers their likely location based on spatiotemporal behavior patterns, rather than direct GPS tracking.

# Table of Contents
- [Overview](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#overview)
- [Objectives](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#objectives)
- [Core Idea](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#core-idea)
- [How the system works](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#how-the-system-works)
- [Project Structure](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#project-structure)
- [Dataset Description](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#dataset-description)
- [System Architecture](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#system-architecture)
- [identity_mobility.csv (Design)](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#identity_mobilitycsv-design)
  - [Schema](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#schema)
    - [Example](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#example)
- [Technologies Used](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#technologies-used)
  - [Python Version Requirement](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#python-version-requirement)
    - [Important](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#important)
    - [Recommended Version](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#recommended-version)
- [How to run](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#how-to-run)
  - [Step 1](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-1-clone-the-repository)
  - [Step 2](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-2-create-and-activate-a-virtual-environment)
    - [Windows](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#windows)
    - [Linux/MacOS](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#linuxmacos)
  - [Step 3](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-3-install-dependencies)
  - [Step 4](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-4-set-up-the-data-directory)
  - [Step 5](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-5-add-raw-face-images)
  - [Step 6](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-6-generate-face-embeddings)
  - [Step 7](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-7-configure-spatiotemporal-behavior)
  - [Step 8](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-8-run-the-real-time-system)
  - [Step 9](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#step-9-controls)
- [Limitations](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#limitations)
- [Ethical Considerations](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#ethical-considerations)
- [Conclusion](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#conclusion)
- [Author](https://github.com/aniketrepo/face-based-location-prediction-system?tab=readme-ov-file#author)

# Overview
This project demonstrates how identity recognition can be combined with rule-based spatiotemporal reasoning to infer a person’s probable location at a given time.

Instead of predicting exact coordinates, the system answers:
 >“Given who this person is, and the current day and time, where are they most likely to be?”

# Objectives
The primary objectives of this project are:
- To implement **real-time face recognition using ArcFace embeddings** for robust and identity-consistent person detection.
- To design a **spatiotemporal behavior model** that represents where individuals typically go based on day and time patterns.
- To **infer a person’s likely location** at a given moment using behavioral rules rather than direct location tracking.
- To **separate identity recognition from location inference** to ensure modularity, 

# Core Idea
The system is built on a clear separation of concerns:
- **Face Recognition**: handeled using ArcFace embeddings
- **Location inference**: handeled using predefined behavioral rules
- **Time awareness**: used to filter plausible locations
- **No tracking**: no GPS, no surveillance, no continous monitoring

# How the system works 
- The webcam continuously captures video frames using OpenCV.
- InsightFace detects faces in each frame and extracts ArcFace embeddings.
- The extracted embedding is compared with stored embeddings using cosine similarity to identify the person.
- If no match crosses the similarity threshold, the face is labeled as `Unknown`.
- For recognized identities, the system looks up predefined spatiotemporal behavior rules.
- The current day and time are used to infer the most likely location.
- Predictions are smoothed over multiple frames to ensure stable output.
- The identity and inferred location are displayed in real time on the webcam feed.

# Project Structure
```shell
face-based-location-prediction-system/
│
├── data/                            # Runtime data (ignored in .gitignore)
│   ├── raw_faces/                   # Raw face images per identity
│   ├── embeddings/                  # ArcFace embeddings (.npy per identity)
│   └── mobility/                    # Spatiotemporal behavior data
│       └── identity_mobility.csv
│
├── build_embeddings.py              # Generates embeddings from raw face images
├── face_utils.py                    # Shared face processing utilities
├── webcam_recognition.py            # Real-time face recognition and location inference
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

# Dataset Description
The project uses a locally curated dataset consisting of face images, face embeddings, and spatiotemporal behavior definitions.
- Raw face images are stored per identity and are used only for generating embeddings.
- Face embeddings are generated using ArcFace and stored as `.npy` files, with one embedding per identity.
- Spatiotemporal behavior data is defined in a CSV file that encodes frequent locations along with valid days, time ranges, and relative weights.

All dataset components are stored locally and excluded from version control to preserve privacy.

The dataset is intended strictly for academic and experimental purposes.

# System Architecture
The system is designed as a modular pipeline with clear separation between perception, recognition, and reasoning components.
- **Input Layer** – Captures live video frames from the webcam.
- **Perception Layer** – Detects faces and extracts ArcFace embeddings using InsightFace.
- **Recognition Layer** – Matches live embeddings against stored identity embeddings using cosine similarity and thresholding.
- **Behavior Knowledge Layer** – Stores spatiotemporal behavior patterns in a structured CSV format.
- **Inference Layer** – Combines recognized identity with current day and time to infer the most likely location.
- **Stabilization Layer** – Applies temporal smoothing and caching to ensure stable predictions.
- **Output Layer** – Displays identity and inferred location in real time on the webcam feed.

# identity_mobility.csv (Design)
Each row represents **one frequent place for one person**, defined by time and day constraints.

## Schema
| Column       | Description                              |
| ------------ | ---------------------------------------- |
| `person_id`  | Identity (must match embedding filename) |
| `place_name` | Semantic location label                  |
| `place_type` | Work / education / personal / leisure    |
| `days`       | Valid days (pipe-separated)              |
| `time_start` | Start of time window                     |
| `time_end`   | End of time window                       |
| `weight`     | Relative frequency (not probability)     |

### Example:
```csv
aniket_dixit,College Campus,education,Mon|Tue|Wed|Thu|Fri,09:00,16:00,0.6
```

# Technologies Used
- **Python**
- **OpenCV** – webcam & visualization
- **InsightFace (ArcFace)** – face detection & embeddings
- **NumPy** – vector operations
- **CSV-based rule engine** – behavioral inference

## Python Version Requirement

This project is tested and verified on **Python 3.10.11**.

### Important

Some libraries used in this project, particularly **InsightFace**, **ONNX Runtime**, and their underlying dependencies, may not be fully compatible with newer Python versions at the time of development.

Using **Python 3.11 or higher** may result in:
- Installation failures for `insightface` or `onnxruntime`
- Runtime errors related to model loading or inference
- Compatibility issues with compiled dependencies

### Recommended Version

- **Python 3.10.11** (tested)

Using a Python version other than the recommended one may lead to unexpected behavior or reduced stability.

# How to run
## Step 1: Clone the repository
```bash
git clone https://github.com/aniketrepo/face-based-location-prediction-system.git
cd face-based-location-prediction-system
```

## Step 2: Create and activate a virtual environment
### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/MacOS
```python
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install dependencies
```python
pip install -r requirements.txt
```

## Step 4: Set up the data directory
Create the following structure manually (this directory is ignored in Git):
```shell
data/
├── raw_faces/
├── embeddings/
└── mobility/
    └── identity_mobility.csv
```
## Step 5: Add raw face images
- Place face images inside `data/raw_faces/`
- Organize images by identity (one folder per person)

Example:
```shell
data/raw_faces/
├── face_1/
├── face_2/
└── face_3/
```

## Step 6: Generate face embeddings
Run the embedding generation script:
```python
python build_embeddings.py
```

This will create one `.npy` embedding file per identity inside:
```shell
data/embeddings/
```

## Step 7: Configure spatiotemporal behavior
Edit the file:
```shell
data/mobility/identity_mobility.csv
```
Ensure:
- `person_id` values exactly match embedding filenames
- Days and time ranges are correctly defined

## Step 8: Run the real-time system
```python
python webcam_recognition.py
```

## Step 9: Controls
- Press `Q` to quit the application

# Output
When the system is running, the webcam window displays real-time face recognition and inferred location information.

On-screen output includes:
- Bounding box around the detected face
- Recognized identity (or `Unknown`)
- Similarity score
- Inferred likely location based on time and behavior
- Quit instruction

# Limitations
- The system relies on predefined spatiotemporal behavior rules and does not learn or adapt behavior dynamically.
- Location inference is probabilistic and based on routine patterns; it does not represent real-time or exact location tracking.
- Face recognition accuracy depends on lighting conditions, camera quality, and the quality of stored embeddings.
- Each identity is represented by a single embedding, which may reduce robustness to pose and appearance variations.
- The system is designed for a limited number of known identities and does not scale efficiently to large databases.
- The system does not handle occlusions, masks, or extreme face angles reliably.

# Ethical Considerations
This project:
- Does **NOT** track real locations
- Does **NOT** store video or biometric data
- Does **NOT** perform surveillance

It performs **probabilistic inference based on predefined behavioral patterns**, intended strictly for academic and experimental use.

# Conclusion
This project combines face recognition with spatiotemporal reasoning to infer a person’s likely location without using real-time tracking. By separating identity recognition from rule-based location inference, the system remains interpretable, efficient, and privacy-aware, making it suitable for academic and experimental use.

## Author
Aniket Dixit  
B.Tech Data Science  
