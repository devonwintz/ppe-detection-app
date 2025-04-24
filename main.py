import os
import sys
import cv2
import time
import numpy as np
from utils.logger import make_logger
from utils.file_loader import load_file
from utils.platform_selector import is_android, get_platform_model

logger = make_logger("app_info", "info")
error_logger = make_logger("app_error", "error")


class PPEApp:
    def __init__(self, config_path):
        self.config = load_file(config_path)
        self.nms_threshold = self.config.get("nmsThreshold", 0.4)
        self.conf = self.config.get("confidenceSource", 0.4)
        self.target_classes = self.config["ppeDetectionClasses"]
        self.frame_skip = self.config.get("frameSkip", 2)

        self.is_android = is_android()
        self.model, self.input_index, self.output_index, self.input_shape = get_platform_model(self.config, self.is_android)

        self.webcam = cv2.VideoCapture(self.config.get("inputSource", 0))
        if not self.webcam.isOpened():
            error_logger.critical("Unable to access the webcam.")
            sys.exit(1)

    def preprocess_image(self, frame):

        if frame is None or frame.size == 0:
            error_logger.error("Empty or invalid frame received; skipping preprocessing.")
            return None

        try:
            resized = cv2.resize(frame, (640, 640))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            error_logger.error(f"Resize failed: {e}")
            return None

        # Optional: Apply CLAHE if enabled
        if self.config.get("enableCLAHE", False):
            try:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                resized = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                error_logger.error(f"CLAHE enhancement failed: {e}")
                return None

        try:
            normalized = resized.astype(np.float32) / 255.0
            input_img = np.expand_dims(normalized, axis=0).astype(np.float32)
        except Exception as e:
            error_logger.error(f"Normalization failed: {e}")
            return None

        if input_img.shape != (1, 640, 640, 3):
            error_logger.error(f"Preprocessed image has invalid shape: {input_img.shape}")
            return None

        return input_img

    def process_frame(self, frame):
        if self.config.get("enablePreprocessing", False):
            frame = self.preprocess_image(frame)
            if frame is None:
                error_logger.warning("Preprocessing failed; skipping frame.")
                return None

        if self.is_android:
            return self.process_with_tflite(frame)
        else:
            return self.process_with_yolo(frame)


    def process_with_yolo(self, frame):
        results = self.model(source=frame, conf=self.conf, show=False, save=False)

        for result in results:
            img = result.orig_img.copy()
            names = result.names

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0])*100

                if cls_name not in self.target_classes:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = f"{cls_name} {conf:.2f}%"

                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return img


    def process_with_tflite(self, frame):
        resized_frame = cv2.resize(frame, (640, 640))
        input_tensor = np.expand_dims(resized_frame, axis=0).astype(np.float32)

        self.model.set_tensor(self.input_index, input_tensor)
        self.model.invoke()

        output_details = self.model.get_output_details()
        output_data = self.model.get_tensor(output_details[0]['index'])  # shape: (1, 14, 8400)
        output_data = np.squeeze(output_data)  # shape: (14, 8400)

        if output_data.shape[0] != 14:
            error_logger.error(f"Unexpected output shape: {output_data.shape}")
            return frame

        img = frame.copy()


        return img


    def run(self):
        i = 0
        frame_count = 0
        last_time = time.time()

        while True:
            ret, frame = self.webcam.read()
            if not ret or frame is None:
                error_logger.error("Failed to capture frame or frame is empty.")
                continue

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            if time.time() - last_time < 0.2:
                continue
            last_time = time.time()

            result_img = self.process_frame(frame)

            if result_img is None:
                continue

            if result_img is None or not isinstance(result_img, np.ndarray):
                logger.info("[ERROR] result_img is not a valid image")
                return

            if self.config.get("showResults", True):
                cv2.imshow("PPE Detection", result_img)

            if self.config.get("saveImagesWithResults", False):
                os.makedirs(self.config["resultsPath"], exist_ok=True)
                cv2.imwrite(f"{self.config['resultsPath']}/result_{i}.jpg", result_img)
                i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = "config/config_android.json" if is_android() else "config/config_pc.json"
    app = PPEApp(config_path)
    app.run()
