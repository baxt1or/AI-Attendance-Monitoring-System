import cv2 as cv
import pickle
import insightface
import numpy as np
import os


app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU (-1), GPU (0 if you have CUDA)


DB_PATH = "../data/embeddings.pkl"


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







# register("test3", "../data/test3.jpg")



# with open(DB_PATH, "wb") as f:
#     pickle.dump(db, f)

# print("\n✅ Done. Stored embeddings:")
# print(list(db.keys()))