from abc import ABC
from abc import abstractmethod


class RuntimeComponent(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
