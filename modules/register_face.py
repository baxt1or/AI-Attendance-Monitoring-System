import cv2 as cv
import pickle
import insightface
import numpy as np
import os
import glob



app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1) 


DB_PATH = "./data/embeddings/embeddings.pkl"


if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
else:
    db = {}


def register(name, image_path):
    img = cv.imread(image_path)

    faces = app.get(img)

    if len(faces) == 0:
        print(f"[ERROR] No face detected in {image_path}")
        return

    if len(faces) > 1:
        print(f"[WARNING] Multiple faces found in {image_path}, using first one")

    embedding = faces[0].embedding

    embedding = embedding / np.linalg.norm(embedding)

    db[name] = embedding

    print(f"[INFO] Registered: {name}")








# SAVES FACE IMAGE DATA
files = glob.glob("../data/students/*")

ids = [os.path.splitext(os.path.basename(file))[0] for file in files]


for student_id, file_path in zip(ids, files):

    print(f"Processing: {student_id}")

    register(student_id, file_path)


with open(DB_PATH, "wb") as f:
    pickle.dump(db, f)

print("\n✅ Done. Stored embeddings:")
print(list(db.keys()))