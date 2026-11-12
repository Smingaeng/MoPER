from logging import getLogger

def log_exceptions(func):
    def wrapper(*args, **kwargs):
        logger = getLogger('train_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper
