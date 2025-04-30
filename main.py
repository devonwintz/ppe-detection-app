import os
import cv2
import time
import random
import numpy as np
from utils.logger import make_logger
from utils.file_loader import load_file
from utils.platform_selector import is_android, get_platform_model

logger = make_logger("app_info", "info")
error_logger = make_logger("app_error", "error")

class PPEApp:
    def __init__(self, config_path):
        self.config = load_file(config_path)
        self.paths = self.config.get("paths", {})
        self.settings = self.config.get("settings", {})
        self.options = self.config.get("options", {})

        self.input_size = self.settings.get("inputSize", 640)
        self.conf = self.settings.get("confidenceThreshold", 0.4)
        self.target_classes = self.config.get("ppeDetectionClasses", [])
        self.frame_skip = self.settings.get("frameSkip", 2)
        self.is_android = is_android()

        self.model, self.input_index, self.output_index, self.input_shape = get_platform_model(self.config, self.is_android)

        self.input_source = self.paths.get("input", 0)
        self.webcam = cv2.VideoCapture(self.input_source)

        if not self.webcam.isOpened():
            error_logger.critical(f"Unable to access the input source: {self.input_source}")
            return

        if self._is_webcam_source():
            if self.is_android:
                try:
                    resolution = self.config.get("settings", {}).get("resolution", [640, 480])
                    width, height = resolution
                    self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    logger.info(f"Android detected: Setting webcam resolution to {width}x{height}")
                except Exception as e:
                    error_logger.error(f"Failed to set webcam resolution: {e}")

            logger.info("Webcam initialized successfully.")
        else:
            logger.info(f"Video/IP stream input '{self.input_source}' initialized successfully.")

        self.light_chance = self.settings.get("lightVariationChance", 0.1)
        self.blur_chance = self.settings.get("blurChance", 0.1)


    def _is_webcam_source(self):
        if isinstance(self.input_source, int):
            return True
        if isinstance(self.input_source, str) and self.input_source.isdigit():
            self.input_source = int(self.input_source)
            return True
        return False


    def apply_random_light_variation(self, frame):
        """
        Randomly adjusts the brightness of the input image to simulate real-world lighting changes.
        """
        if random.random() < self.light_chance:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
            value = random.randint(-30, 30)  # Random light adjustment between -30 and +30
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)  # Change the brightness (V channel)
            hsv = hsv.astype(np.uint8)
            frame =  cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame


    def apply_random_blur(self, img):
        """
        Randomly applies a Gaussian blur to the input image to simulate real-world motion or camera instability.
        """
        if random.random() < self.blur_chance:
            ksize = random.choice([3, 5, 7])  # Randomly choose a kernel size
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img


    def preprocess_image(self, frame):
        is_preprocessed  = False

        if frame is None or frame.size == 0:
            error_logger.error("Empty or invalid frame received; skipping preprocessing.")
            return None, is_preprocessed

        try:
            resized = cv2.resize(frame, (self.input_size, self.input_size))
            if self.options.get("applyLightVariation", False):
                resized = self.apply_random_light_variation(resized)
            if self.options.get("applyBlur", False):
                resized = self.apply_random_blur(resized)
            is_preprocessed  = True
        except cv2.error as e:
            error_logger.error(f"Resize failed: {e}")
            return None, is_preprocessed

        # Optional: Apply CLAHE if enabled
        if self.options.get("enableCLAHE", False):
            try:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                resized = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                error_logger.error(f"CLAHE enhancement failed: {e}")
                return None, is_preprocessed

        try:
            normalized = resized.astype(np.float32) / 255.0
            preprocessed_frame = np.expand_dims(normalized, axis=0).astype(np.float32)
        except Exception as e:
            error_logger.error(f"Normalization failed: {e}")
            return None, is_preprocessed

        if preprocessed_frame.shape != (1, 640, 640, 3):
            error_logger.error(f"Preprocessed image has invalid shape: {preprocessed_frame.shape}")
            return None, is_preprocessed

        return preprocessed_frame, is_preprocessed


    def process_frame(self, frame):
        is_preprocessed = False

        if self.options.get("enablePreprocessing", False):
            preprocessed_frame, is_preprocessed = self.preprocess_image(frame)
            if preprocessed_frame is None:
                error_logger.warning("Preprocessing failed; skipping frame.")
                return None
        else:
            preprocessed_frame = frame
        if self.is_android:
            return self.process_with_tflite(preprocessed_frame, is_preprocessed)
        else:
            return self.process_with_yolo(preprocessed_frame, is_preprocessed)


    def process_with_yolo(self, frame, is_preprocessed=False):
        if is_preprocessed:
            img = frame
            if img.shape[0] == 1:
                img = img.squeeze(axis=0)  # Remove the first dimension to get (640, 640, 3)
            img = (img * 255).astype(np.uint8)  # De-normalize frame for YOLO
        else:
            resized = cv2.resize(frame, (self.input_size, self.input_size))
            img = resized

        # Check if img has the expected shape (h, w, c)
        if img is None or len(img.shape) != 3:
            error_logger.error(f"Invalid image shape: {img.shape if img is not None else 'None'}")
            return None

        results = self.model(source=img, conf=self.conf, show=False, save=False)

        for result in results:
            img = result.orig_img.copy()
            names = result.names

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0]) * 100

                if cls_name not in self.target_classes:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = f"{cls_name} {conf:.2f}%"

                color = (0, 255, 0) if "NO-" not in cls_name else (0, 0, 255)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img



    def process_with_tflite(self, frame, is_preprocessed):
        if is_preprocessed:
            input_tensor = frame
            frame = (frame[0] * 255).astype(np.uint8)  # De-normalize frame for drawing
        else:
            frame = cv2.resize(frame, (self.input_size, self.input_size))
            input_tensor = frame.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

        self.model.set_tensor(self.input_index, input_tensor)
        self.model.invoke()
        output = self.model.get_tensor(self.output_index)

        detections = output[0]

        h, w = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2, conf, cls_id = detection
            conf = float(conf)
            if conf < self.conf:
                continue

            cls_id = int(cls_id)
            cls_name = self.target_classes[cls_id] if cls_id < len(self.target_classes) else f"Class {cls_id}"
            if cls_name not in self.target_classes:
                continue

            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            label = f"{cls_name} {conf * 100:.1f}%"
            color = (0, 255, 0) if "NO-" not in cls_name else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def run(self):
        i = 0
        frame_count = 0

        while True:
            ret, frame = self.webcam.read()
            if not ret or frame is None:
                error_logger.critical("Failed to capture frame or frame is empty.")
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            result_img = self.process_frame(frame)
            # Simulate mobile delay
            if not self.is_android:
                time.sleep(0.05)

            if result_img is None:
                continue

            if not isinstance(result_img, np.ndarray):
                logger.error("Image is not a valid image")
                break

            if self.options.get("showResults", True):
                cv2.imshow("PPE Detection", result_img)

            if self.options.get("saveImagesWithResults", False):
                os.makedirs(self.config["paths"]["results"], exist_ok=True)
                cv2.imwrite(f"{self.config['paths']['results']}/result_{i}.jpg", result_img)
                i += 1

            if cv2.waitKey(self.settings.get("frameDelayMilliseconds", 100)) & 0xFF == ord('q'):
                break

        self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = "config/config_android.json" if is_android() else "config/config_pc.json"
    app = PPEApp(config_path)
    app.run()
