# human-pose-estimation
# Muscle Outline Detection using MediaPipe and OpenCV

This project detects human body joints and highlights muscle regions by drawing lines over arms, legs, and the torso using pose estimation.

## 🔧 Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Place an image named `person.jpg` in the `input/` folder.

3. Run the main script:

```bash
python main.py
```

## 📂 Output

The processed image will be saved in the `output/` folder as `pose_output.jpg`.

## ✅ Requirements

- Python 3.8+
- OpenCV
- MediaPipe
