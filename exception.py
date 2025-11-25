from src.logger import logger

class CustomException(Exception):
    def __init__(self, message, errors=None):
        self.message = message
        self.errors = errors
        logger.error(f"CustomException: {self.message} | Errors: {self.errors}")
    
    def __str__(self):
        return f"{self.message} | Errors: {self.errors}"
   

if __name__ == "__main__":
    try:
        logger.info("CustomException has been raised and logged.")
        raise CustomException("This is a custom exception", {"code": '!200', "detail": "Please doucble check your code logic"})
    except CustomException as ce:
        logger.exception("An exception occurred")
    
    