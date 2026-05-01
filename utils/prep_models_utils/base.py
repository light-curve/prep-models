from abc import ABC, abstractmethod
from pathlib import Path


class ModelAdapter(ABC):
    @abstractmethod
    def fetch(self, *, update: bool = False) -> None:
        """Initialize the pinned submodule, or update it to upstream when requested."""

    @abstractmethod
    def download(self) -> None:
        """Download pretrained weights into models/<name>/weights/."""

    @abstractmethod
    def export(self, output_dir: Path) -> None:
        """Export model to ONNX files (one per aggregation variant) in output_dir."""

    @abstractmethod
    def test_data(self, output_dir: Path, n_samples: int = 5) -> None:
        """Run original preprocessing + model on real (or synthetic) data, save parquet."""
