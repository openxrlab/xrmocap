import logging

# client does not require xrprimer.utils.log_utils
# logger's level is set to INFO by default
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
