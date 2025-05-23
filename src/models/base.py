from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def inference(self, user_id: int):
        pass
