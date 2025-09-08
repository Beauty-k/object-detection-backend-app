
class DetectionError(Exception):
    """
    Raised when an object detection operation fails.

    Attributes:
        message (str): readable explanation of the error.
        cause (Exception): Optional underlying exception that caused this error.
    """

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause