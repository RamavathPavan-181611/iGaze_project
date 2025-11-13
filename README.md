# 👁️ iGaze — Eye-Gaze and Blink-Based Control System

**iGaze** is an AI-powered human–computer interaction system that enables hands-free control using only eye movements and blinks.  
It leverages **MediaPipe FaceMesh**, **OpenCV**, and **PyAutoGUI** to provide intuitive gaze tracking and blink-based clicking.

---

## 🚀 Features
- Real-time **eye gaze tracking** for cursor movement  
- **Blink detection** for left, right, and double clicks  
- **Voice control** for optional speech commands  
- **Calibration** to improve accuracy and comfort  
- **Virtual Keyboard** (Tkinter-based) for text input  

---

## 🧠 Tech Stack
| Component | Library / Tool |
|------------|----------------|
| Eye Tracking | MediaPipe FaceMesh |
| Cursor Control | PyAutoGUI |
| Computer Vision | OpenCV |
| GUI Keyboard | Tkinter |
| Voice Commands | SpeechRecognition |
| Language | Python 3.x |

---
## Working and condition:
- short time blink less than 0.5 are ignored  (i.e time in sec)
- right blink (0.5 - 1) = left click
- right blink (1 - 2) = scroll down
- left blink (2 - 3) = right click
- left blink (more than 3) = scroll

- voice recognition:(click, double click, scroll up, scroll down, open chrome, right click, open , enter, click two times, ..)
  

## 🧩 Installation
```bash
git clone https://github.com/yourusername/iGaze.git
cd iGaze
pip install -r requirements.txt
