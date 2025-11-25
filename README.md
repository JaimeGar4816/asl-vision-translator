# ASL Vision Translator  
**Real-Time American Sign Language Recognition using Computer Vision + Machine Learning + Motion Tracking**

Welcome to the ASL Vision Translator — a real-time American Sign Language (ASL) letter recognition system that uses MediaPipe, Python, and a custom-trained machine learning model.  
This project also supports **dynamic motion tracking** for letters **J** and **Z**, smart word suggestions, and an intuitive UI for real-time communication.

This system was built with the goal of creating accessible, intelligent technology for ASL users and learners — while showcasing real engineering, machine learning, and computer vision skills.

---

## Features

### **Real-Time ASL Recognition**
- Recognizes static ASL letters **A–I, K–Y** using ML
- Uses hand landmark extraction (63 features from MediaPipe)
- Stable-prediction filtering to reduce noise

### **Dynamic Motion Tracking (J & Z)**
ASL letters **J** and **Z** require movement.  
The system:
- Tracks index fingertip motion  
- Computes motion path length  
- Classifies J/Z from drawn motion  

### **Smart Word Suggestions**
- System predicts likely words as you sign  
- Supports top 1–3 suggestions  
- Accept suggestions with keys `1`, `2`, or `3`

### **Live UI Panel**
- Shows current prediction  
- Shows built text  
- Shows motion mode  
- Shows word suggestions  

### **Dataset Collection Tool**
- Collect custom ASL samples  
- Append mode (won’t delete old data)  
- Visual markers  
- Per-letter sample counts  

### **Model Training Script**
- Random Forest classifier  
- Label encoding  
- Train/test split  
- Accuracy + classification report output  
- Saves model to `/models/asl_model.joblib`

---

## Project Structure

asl-vision-translator/
│
├── asl-vision-translator/
│ ├── data/
│ │ └── asl_data.npz
│ │
│ ├── models/
│ │ └── asl_model.joblib
│ │
│ ├── src/
│ │ ├── collect_asl_data.py
│ │ ├── train_asl_model.py
│ │ └── asl_live_demo.py
│
├── .gitignore
├── LICENSE
└── README.md

---

## Installation

Clone the repository:
git clone https://github.com/JaimeGar4816/asl-vision-translator.git
cd asl-vision-translator

Install dependencies:
pip install opencv-python mediapipe scikit-learn joblib numpy

## Steps
Step 1
Run:
python3 src/collect_asl_data.py

Controls:
A–Z → choose letter
6 → save sample
1 → quit and save
Shows sample counts
Data saved to:
data/asl_data.npz

Step 2 — Train the Model
Run:
python3 src/train_asl_model.py

Outputs:
Accuracy score
Classification report
Saved model:
models/asl_model.joblib

Step 3 — Run the Live Demo
Run:
python3 src/asl_live_demo.py

-----Static Letters
Hold your hand in a letter shape.
Prediction appears when stable.
-----Motion Letters (J & Z)
0 → enable motion mode
J / Z → choose letter
Move fingertip to draw motion
M → finish and commit letter
----Text Editing
space → add space
b → backspace
c → clear text
q → quit
----Word Suggestions
Appears automatically
Press 1, 2, or 3 to insert suggestion

### License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
This means:
- You may view and modify the code
- You must credit the creator (Jaime Garcia)
- You may NOT use it commercially
- You may NOT sell or profit from this work
- Any modifications must use the same license
Jaime Garcia retains full commercial rights and may license or sell the work.

### Future Improvements (Roadmap)
- Full ASL word recognition
- Multi-hand support
- Depth-based gesture detection
- Real-time sentence translation
- Mobile app version (iOS & Android)
- Spanish translation support

### Author
Jaime Garcia
Developer — ASL Vision Translator
Computer Science Student | Machine Learning | Computer Vision
If you'd like to collaborate, extend the project, or request commercial usage, feel free to reach out.

Support the Project
If you like this project, give the repo a star ⭐ — it means a lot and helps others discover it!
