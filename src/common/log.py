import logging
import sys


class Log(object):

    """Logger"""

    @staticmethod
    def set_logger(logger_name: str):
        """create logging object.

        :param logger_name: Name of logger (present in logging).
        """
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(c_handler)
        return logger


if __name__ == "__main__":
    pass
