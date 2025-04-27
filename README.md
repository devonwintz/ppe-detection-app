# A Portable, Lightweight PPE Compliance Detection App

This is a lightweight computer vision system for PPE (Personal Protective Equipment) compliance detection, designed for transient and resource-limited worksites. This application enables quick setup for entry-point monitoring and runs efficiently on edge devices like Raspberry Pi or Android smartphones—requiring only onboard processing, limited memory, and minimal storage.

### 1. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Convert to TensorFlow Lite (Optional)
Have a custom PyTorch model? Simply add it to the `models/` directory and run the following command to convert it to ONNX and TensorFlow Lite formats for lightweight deployment.

```bash
python convert/yolov82tflite.py
```

### 3. Run the App
```bash
python main.py
```
<br/>