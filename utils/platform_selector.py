import platform
import os

def is_android():
    # Defaults to Android (True). Remove 'not' for dynamic platform selection
    return not (platform.system() == "Linux" and "ANDROID_STORAGE" in os.environ)

def get_platform_model(config, is_android):
    if is_android:
        from tensorflow.lite.python.interpreter import Interpreter
        model = Interpreter(model_path=config["paths"]["model"])
        model.allocate_tensors()

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_index = input_details[0]['index']
        output_index = output_details[0]['index']
        input_shape = input_details[0]['shape'][1:3]

        return model, input_index, output_index, input_shape
    else:
        from ultralytics import YOLO
        model = YOLO(config["paths"]["model"])
        return model, None, None, None