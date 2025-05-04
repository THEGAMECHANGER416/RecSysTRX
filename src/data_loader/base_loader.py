from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass
