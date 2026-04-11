# 🤖 AI Attendance Monitoring System

A real-time face recognition-based attendance system powered by AI and computer vision.

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 🧠 Face recognition using embeddings (InsightFace)
* 📊 Automatic attendance logging to CSV
* ⚡ Fast cosine similarity matching
* 🧍 Multi-person detection support
* 🔁 Prevents duplicate attendance entries

## 🧠 How It Works

1. Face embeddings are precomputed and stored in a database (`embeddings.pkl`)
2. The webcam captures live video
3. Faces are detected and converted into embeddings
4. Cosine similarity is used to match faces with known identities
5. Recognized individuals are marked present in a CSV file

## 🏗️ Tech Stack

* Python
* OpenCV (`cv2`)
* InsightFace
* NumPy
* Pickle (for embedding storage)

## 📂 Project Structure

```id="s7y1xp"
.
├── main.py                          # Main attendance system
├── data/
│   └── embeddings/
│       └── embeddings.pkl          # Stored face embeddings
├── attendance/
│   └── attendance_YYYY-MM-DD.csv   # Daily attendance logs
└── requirements.txt
```

## ⚙️ Setup

### 1. Clone repository

```bash id="h8n9qx"
git clone https://github.com/your-username/ai-face-attendance.git
cd ai-face-attendance
```

### 2. Install dependencies

```bash id="y2v3lm"
pip install -r requirements.txt
```

### 3. Run the system

```bash id="t9k4jb"
python main.py
```

## 📌 Example Output

```id="xw5m1z"
[ATTENDANCE] John marked present!
[ATTENDANCE] Alice marked present!
```

Generated CSV:

```id="m8v2qp"
2026-04-11 09:01:12,John
2026-04-11 09:02:45,Alice
```

## ⚙️ Configuration

* **Threshold tuning** (similarity):

```python id="k2d9fa"
THRESHOLD = 0.65
```

* Lower → more matches (risk of false positives)
* Higher → stricter matching

## 🔮 Future Improvements

* 📱 Web dashboard for attendance tracking
* 🧠 Face registration pipeline (UI)
* ☁️ Cloud deployment (AWS / GCP)
* 🧾 Database integration (PostgreSQL)
* 🔐 Anti-spoofing (liveness detection)
* 🎯 Improve accuracy with multiple embeddings per user

## ⚠️ Limitations

* Sensitive to lighting conditions
* Requires precomputed embeddings
* CPU mode may be slower for large groups

## 👤 Authors

**Kamila Atahodjayeva**
Data Scientist | Computer Vision Engineer

**Baxtiyor Bekmurodov**
Data Scientist | Computer Vision Engineer

---

⭐ Star this repo if you find it useful!
