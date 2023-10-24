import logging
import os

if os.environ.get("log_logger_folder_name") == None:
    log_logger_folder_name = "logs_default"
    if not os.path.exists(f"./{log_logger_folder_name}"):
        os.makedirs(f"./{log_logger_folder_name}")
else:
    log_logger_folder_name = os.environ.get("log_logger_folder_name")
    if not os.path.exists(log_logger_folder_name):
        os.makedirs(log_logger_folder_name)

logger = logging.getLogger(__name__)

if os.environ.get("log_file_name") == None:
    log_name = f"{log_logger_folder_name}/test.log"
else:
    log_name = f"{log_logger_folder_name}/" + os.environ.get("log_file_name")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    filename=log_name, filemode='w')
