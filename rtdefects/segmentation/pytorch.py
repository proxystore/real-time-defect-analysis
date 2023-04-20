"""Implementation using PyTorch.Segmentation"""
from hashlib import md5
from pathlib import Path
from typing import Optional
import logging

import segmentation_models_pytorch as smp

import albumentations as albu
from skimage import color
from skimage.transform import resize
import numpy as np
import requests
import torch

from rtdefects.segmentation import BaseSegmenter

logger = logging.getLogger(__name__)

# Storage for the model
_model: Optional[torch.nn.Module] = None
_model_dir = Path(__file__).parent.joinpath('files')

# Lookup tables for the pre-processor used by different versions of the model
_encoders = {
    'voids_segmentation_091321.pth': 'se_resnext50_32x4d',
    'voids_segmentation_030323.pth': 'efficientnet-b3',
    'small_voids_031023.pth': 'se_resnext50_32x4d',
}


def download_model(name: str):
    """Download a model to local storage

    Args:
        Name of the model
    """
    my_path = _model_dir / name
    with requests.get(f"https://g-29c18.fd635.8443.data.globus.org/ivem/models/{name}", stream=True) as r:
        r.raise_for_status()
        with open(my_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


class PyTorchSegmenter(BaseSegmenter):
    """Interfaces for models based on segmentation_models.pytorch"""

    def __init__(
            self,
            model_name: str = 'small_voids_031023.pth',
    ):
        """
        Args:
            model_name: Name of the model we should use
        """

        # Get the preprocessor and build the preprocessing pipeline
        assert model_name in _encoders, f'No encoder defined for {model_name}. Consult developer'
        preprocessing_fn = smp.encoders.get_preprocessing_fn(_encoders[model_name])

        # Store the path to the model
        self.model_path = _model_dir / model_name
        if not self.model_path.is_file():
            logger.info('Downloading model')
            download_model(model_name)
        assert self.model_path.is_file(), 'Download failed'

        # Define the conversion from image to inputs
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        self.preprocess = albu.Compose(_transform)

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        # Convert to RGB
        image: np.ndarray = color.gray2rgb(image_data)

        # Scale to 1024x1024
        if image.shape[:2] != (1024, 1024):
            image = resize(image, output_shape=(1024, 1024), anti_aliasing=True)

        # Perform the preprocessing
        image = self.preprocess(image=image)

        return image['image']

    def _load_model(self, device: str):
        global _model
        if _model is None:
            # Make sure the model exists
            if not self.model_path.is_file():
                raise ValueError(f'Cannot find the model. No such file: {self.model_path}')

            # Get the model hash to help with reproducibility
            with open(self.model_path, 'rb') as fp:
                hsh = md5()
                while len(line := fp.read(4096 * 1024)) > 0:
                    hsh.update(line)
            logger.info(f'Loading the model from {self.model_path}. MD5 Hash: {hsh.hexdigest()}')

            # Load it with Keras
            _model = torch.load(str(self.model_path), map_location=device)
            logger.info('Model loaded.')
        return _model

    def perform_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        # Determine the device at runtime
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self._load_model(device)

        # Push the image to device
        x_tensor = torch.from_numpy(image_data).to(device).unsqueeze(0)

        # Run prediction and get it back from the CPU
        pr_mask = model.predict(x_tensor)
        mask = pr_mask.squeeze().cpu().numpy()

        return mask
