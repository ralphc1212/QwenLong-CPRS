class InvalidInputError(KeyError):
    """Raised when the input is invalid"""
    pass

class SchemaError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class BackendError(Exception):
    def __init__(self, error_body):
        super().__init__(error_body)
        self.error_body = error_body
