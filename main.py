import os
import time
import cv2
import psutil
import random
import numpy as np
from utils.logger import make_logger
from utils.file_loader import load_file
from utils.boxes_overlap import boxes_overlap
from utils.evaluate_predictions import evaluate_all_frames
from utils.platform_selector import is_android, get_platform_model

logger = make_logger("app_info", "info")
error_logger = make_logger("app_error", "error")

start_time = time.time()

class PPEApp:
    def __init__(self, config_path):
        self.config = load_file(config_path)
        self.paths = self.config.get("paths", {})
        self.settings = self.config.get("settings", {})
        self.options = self.config.get("options", {})

        self.input_size = self.settings.get("inputSize", 640)
        self.image_inference_mode = self.options.get("imageInferenceMode", False)
        self.images_path = self.paths.get("inputImages", "./data/images/")
        self.output_label_path = self.paths.get("outputLabels", "./data/test/")
        self.conf = self.settings.get("confidenceThreshold", 0.4)
        self.target_classes = self.config.get("ppeDetectionClasses", [])
        self.frame_skip = self.settings.get("frameSkip", 2)
        self.is_android = is_android()
        self.total_inference_time = 0.0
        self.total_frames_processed = 0

        self.model, self.input_index, self.output_index, self.input_shape = get_platform_model(self.config, self.is_android)

        self.input_source = self.paths.get("input", 0)
        self.webcam = cv2.VideoCapture(self.input_source)

        if not(self.options.get("imageInferenceMode", False)):
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


    def process_with_yolo(self, frame, image_name, is_preprocessed=False):
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

        start_inference = time.time()
        results = self.model(source=img, conf=self.conf, show=False, save=False)
        inference_duration = time.time() - start_inference
        self.total_inference_time += inference_duration
        self.total_frames_processed += 1

        image_width, image_height = img.shape[1], img.shape[0]

        if self.image_inference_mode:
            with open(f"{self.output_label_path}/{image_name}.txt", "w") as f:
                for result in results:
                    names = result.names

                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = names[cls_id]
                        conf = float(box.conf[0])

                        # Normalize the bounding box coordinates to be relative to the image size
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x_center = (xyxy[0] + xyxy[2]) / 2 / image_width
                        y_center = (xyxy[1] + xyxy[3]) / 2 / image_height
                        width = (xyxy[2] - xyxy[0]) / image_width
                        height = (xyxy[3] - xyxy[1]) / image_height

                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
        else:
            names = self.model.names
            person_boxes = []
            violation_boxes = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

                for i in range(len(boxes)):
                    cls_id = int(class_ids[i])
                    cls_name = names[cls_id]
                    conf = confidences[i]
                    xyxy = boxes[i].astype(int)

                    if cls_name == "Person":
                        person_boxes.append((xyxy, conf))
                    elif cls_name in ["NO-Hardhat", "NO-Safety Vest"]:
                        violation_boxes.append((xyxy, cls_name, conf))

            # Now assign compliance status
            for p_box, p_conf in person_boxes:
                status = "Compliant"
                for v_box, v_name, v_conf in violation_boxes:
                    if boxes_overlap(p_box, v_box, threshold=0.3):
                        status = "Non-Compliant"
                        break

                label = f"{status}"
                color = (0, 0, 255) if status == "Non-Compliant" else (0, 255, 0)
                cv2.rectangle(img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
                cv2.putText(img, label, (p_box[0], p_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Optionally display violations too
            for v_box, v_name, v_conf in violation_boxes:
                cv2.rectangle(img, (v_box[0], v_box[1]), (v_box[2], v_box[3]), (0, 255, 255), 2)
                cv2.putText(img, f"{v_name} {v_conf:.2f}%", (v_box[0], v_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return img


    def process_with_tflite(self, frame, is_preprocessed):
        if is_preprocessed:
            input_tensor = frame
            frame = (frame[0] * 255).astype(np.uint8)  # De-normalize frame for drawing
        else:
            frame = cv2.resize(frame, (self.input_size, self.input_size))
            input_tensor = frame.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

        start_inference = time.time()
        self.model.set_tensor(self.input_index, input_tensor)
        self.model.invoke()
        output = self.model.get_tensor(self.output_index)
        inference_duration = time.time() - start_inference
        self.total_inference_time += inference_duration
        self.total_frames_processed += 1

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


    def run_inference_on_images(self, start_cpu_usage, start_ram_usage):
        counter = 0
        images = os.listdir(self.images_path)
        os.makedirs(self.output_label_path, exist_ok=True)

        for image in images:
            counter += 1
            image_path = os.path.join(self.images_path, image)
            img = cv2.imread(image_path)
            if img is None:
                error_logger.warning(f"Unable to read image: {image_path}")
                continue

            image_name = os.path.splitext(os.path.basename(image))[0]
            if self.is_android:
                pass
            else:
                self.process_with_yolo(img, os.path.basename(image_name))

        # System Resource Usage
        system_usage_info = self.system_usage(start_cpu_usage, start_ram_usage)
        logger.info("System Resource Usage: %s", system_usage_info)
        logger.info(f"Inference completed on {counter} images. Results stored at '{self.output_label_path}'")

        # End time for execution
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        logger.info(f"Total execution time: {elapsed_seconds:.2f} seconds")
        logger.info(f"Average inference time per image: {self.get_average_inference_time() * 1000:.2f} ms")

        # Simulated FPS
        simulated_fps = 1.0 / self.get_average_inference_time()
        logger.info(f"Simulated Frame Rate (FPS): {simulated_fps:.2f} seconds")

        # Accuracy Metrics
        accuracy_metrics = evaluate_all_frames(
            frame_dir=self.paths.get("inputImages", "./data/images/"),
            gt_dir=self.paths.get("groundTruthLabels", "./data/labels/ground_truth"),
            pred_dir=self.paths.get("outputLabels", "./data/labels/predictions")
        )
        logger.info(f"True Positives: {accuracy_metrics['TP']}")
        logger.info(f"False Positives: {accuracy_metrics['FP']}")
        logger.info(f"False Negatives: {accuracy_metrics['FN']}")
        logger.info(f"Precision: {accuracy_metrics['precision']}")
        logger.info(f"Recall: {accuracy_metrics['recall']}")
        logger.info(f"F1-Score: {accuracy_metrics['f1Score']}")
        logger.info(f"False Positive Per Class: {accuracy_metrics['FPperClass']}")
        logger.info(f"False Negative Per Class: {accuracy_metrics['FNperClass']}")


    def system_usage(self, start_cpu_usage, start_ram_usage):
        end_cpu_usage = psutil.cpu_percent(interval=1)
        end_ram_info = psutil.virtual_memory()
        end_ram_usage = end_ram_info.used / (1024 ** 2)

        system_usage = {
            'cpuUsage': f"{(end_cpu_usage - start_cpu_usage):.2f}%",
            'ramUsage': f"{(end_ram_usage - start_ram_usage):.2f} MB"
        }

        return system_usage


    def get_average_inference_time(self):
        if self.total_frames_processed == 0:
            return 0.0
        return self.total_inference_time / self.total_frames_processed


    def run(self, start_cpu_usage, start_ram_usage):
        if self.image_inference_mode:
            logger.info("Running inference on images...")
            self.run_inference_on_images(start_cpu_usage, start_ram_usage)
            return

        i = 0
        frame_count = 0

        while True:
            ret, frame = self.webcam.read()
            if not ret or frame is None:
                error_logger.critical("Failed to capture frame or frame is empty.")
                system_usage_info = self.system_usage(start_cpu_usage, start_ram_usage)
                logger.info("System Resource Usage: %s", system_usage_info)
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

        # End time for execution
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        logger.info(f"Total execution time: {elapsed_seconds:.2f} seconds")
        logger.info(f"Average inference time per frame: {self.get_average_inference_time() * 1000:.2f} ms")

        # Simulated FPS
        simulated_fps = 1.0 / self.get_average_inference_time()
        logger.info(f"Simulated Frame Rate (FPS): {simulated_fps:.2f} seconds")


if __name__ == "__main__":
    config_path = "config/config_android.json" if is_android() else "config/config_pc.json"
    app = PPEApp(config_path)

    start_cpu_usage = psutil.cpu_percent(interval=1)
    start_ram_usage = (psutil.virtual_memory().used) / (1024 ** 2)

    app.run(start_cpu_usage, start_ram_usage)
