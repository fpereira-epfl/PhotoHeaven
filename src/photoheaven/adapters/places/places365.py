"""Places365 scene-classification adapter.

Downloads the pre-trained Places365 CNN weights and category labels on first
use and caches them under ``~/.cache/photoheaven/places365``.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_DEFAULT_ARCH = "resnet50"

_MODEL_URLS: dict[str, str] = {
    "resnet18": "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
    "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
}

_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
)


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "photoheaven" / "places365"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", url, dest)
    urllib.request.urlretrieve(url, dest)


def _ensure_file(url: str, dest: Path) -> None:
    if not dest.exists():
        _download(url, dest)


class Places365SceneIdentifier:
    """Identify the scene of an image using a Places365 CNN.

    The classifier returns ``(label, confidence)`` when the top prediction's
    probability is above *threshold*, otherwise ``(None, confidence)``.
    """

    def __init__(self, arch: str = _DEFAULT_ARCH, threshold: float = 0.1) -> None:
        self.arch = arch
        self.threshold = threshold
        self._model: torch.nn.Module | None = None
        self._labels: list[str] = []
        self._device: str | None = None

    @property
    def version(self) -> str:
        return f"places365-{self.arch}-t{self.threshold}"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            import torchvision.models as models
        except ImportError as exc:
            raise RuntimeError(
                "Places365 scene classification requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = _cache_dir()

        model_file = cache / f"{self.arch}_places365.pth.tar"
        _ensure_file(_MODEL_URLS[self.arch], model_file)

        labels_file = cache / "categories_places365.txt"
        _ensure_file(_CATEGORIES_URL, labels_file)

        self._labels = self._load_labels(labels_file)

        model = models.__dict__[self.arch](num_classes=365)
        checkpoint = torch.load(
            model_file,
            map_location="cpu",
            weights_only=False,
            encoding="latin1",
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)
        self._model = model

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        labels: list[str] = []
        with path.open() as class_file:
            for line in class_file:
                parts = line.strip().split()
                if not parts:
                    continue
                # Lines look like "/a/airfield 0"; keep the path after the
                # leading class prefix and turn underscores into spaces.
                label = parts[0].split("/", 2)[-1].replace("_", " ")
                labels.append(label)
        return labels

    def identify(self, image_path: Path) -> tuple[str | None, float | None]:
        """Return the scene label and confidence for *image_path*."""
        self._load_model()

        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        with Image.open(image_path) as img:
            rgb_image = img.convert("RGB")

        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        input_tensor = preprocess(rgb_image).unsqueeze(0).to(self._device)

        assert self._model is not None
        with torch.no_grad():
            logits = self._model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            confidence, index = probabilities.max(0)

        confidence_value = float(confidence)
        if confidence_value < self.threshold:
            return None, confidence_value
        return self._labels[int(index)], confidence_value
