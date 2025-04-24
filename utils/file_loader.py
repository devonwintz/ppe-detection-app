import os
import sys
import json
import csv
from utils.logger import make_logger

logger = make_logger("app_info", "info")
error_logger = make_logger("app_error", "error")

def load_file(file_path):
    if not os.path.exists(file_path):
        error_logger.critical(f"File does not exist: '{file_path}'")
        sys.exit(1)

    _, ext = os.path.splitext(file_path.lower())

    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                logger.info(f"Successfully loaded '{os.path.basename(file_path)}'")
                return f.read()

        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                logger.info(f"Successfully loaded '{os.path.basename(file_path)}'")
                return json.load(f)

        elif ext == '.csv':
            with open(file_path, newline='', encoding='utf-8') as f:
                logger.info(f"Successfully loaded '{os.path.basename(file_path)}'")
                return list(csv.reader(f))

        else:
            error_logger.critical(f"Unsupported file extension: '{ext}'")
            sys.exit(1)

    except Exception as e:
            error_logger.exception(f"Failed to load '{file_path}': {e}")
            sys.exit(1)
