from types import SimpleNamespace
from typing import Callable, Type


class ReaderSpec(SimpleNamespace):
    def __init__(self, data_format: str, data_class: Type, function: Callable) -> None:
        self.data_format = data_format
        self.data_class = data_class
        self.function = function
