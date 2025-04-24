import platform
import os

def is_android():
    # return platform.system() == "Linux" and "ANDROID_STORAGE" in os.environ
    return True

def get_platform_model(config, is_android):
    if is_android:
        # from tflite_runtime.interpreter import Interpreter
        from tensorflow.lite.python.interpreter import Interpreter
        model = Interpreter(model_path=config["modelPath"])
        model.allocate_tensors()

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_index = input_details[0]['index']
        output_index = output_details[0]['index']
        input_shape = input_details[0]['shape'][1:3]

        return model, input_index, output_index, input_shape
    else:
        from ultralytics import YOLO
        model = YOLO(config["modelPath"])
        return model, None, None, None