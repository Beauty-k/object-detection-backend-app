from abc import ABC, abstractmethod

class ICalculator(ABC):
    """
    Interface for calculator classes.

    Defines a common contract with a `calculate` method
    that must be implemented by subclasses.
    """
    
    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass
    