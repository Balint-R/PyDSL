from enum import auto, Enum

class MethodType(Enum):
    """
    abc
    """
    INSTANCE = auto()
    """
    def
    """
    CLASS = auto()
    STATIC = auto()
    CLASS_ONLY = auto()

print(MethodType.__doc__)
print(MethodType.INSTANCE.__doc__)
print(MethodType.CLASS.__doc__)
