class DomainException(Exception):
    def __init__(self, message: str):
        self.message = message

class AIProviderError(DomainException):
    pass

class InvalidMessageError(DomainException):
    pass