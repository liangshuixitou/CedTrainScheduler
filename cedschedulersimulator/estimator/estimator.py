from abc import ABC
from abc import abstractmethod


class Estimator(ABC):
    """
    Estimator is an abstract class that defines train job running duration and priority.
    """

    @abstractmethod
    def inference(self, *args, **kwargs):
        raise NotImplementedError
