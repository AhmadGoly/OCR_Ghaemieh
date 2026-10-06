from abc import ABC, abstractmethod
from PIL import Image

class BaseOCRModel(ABC):
    @abstractmethod
    def process(self, image: Image.Image, lang: str = None) -> str:
        """
        Processes the given image and returns the extracted text.
        """
    def ping(self) -> dict:
        """
        Health probe check for the backend model.
        Returns a dict with 'status': 'online', 'error': None, or detailed issue.
        """
        return {"status": "online", "type": "local"}

    def __repr__(self):
        return f"{self.__class__.__name__}()"
