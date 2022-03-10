import sys
import logging


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter('%(asctime)s|%(filename)s[%(lineno)d]|%(message)s')

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
