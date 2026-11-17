from abc import ABC, abstractmethod
from pathlib import Path


class BaseTagger(ABC):
    def __init__(self, cfg: dict, model_key: str):
        self.cfg = cfg
        self.model_key = model_key
        self.mcfg = cfg["models"][model_key]

    @abstractmethod
    def train(self, train_sents: list, dev_sents: list, test_sents: list) -> None:
        pass

    @abstractmethod
    def load(self, artifact_path: Path) -> "BaseTagger":
        pass

    @abstractmethod
    def __call__(self, text: str) -> list[dict]:
        pass