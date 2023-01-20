"""Implementation using PyTorch.Segmentation"""
from hashlib import md5
from pathlib import Path
from typing import Optional
import logging

import segmentation_models_pytorch as smp
from skimage import color
import albumentations as albu
import numpy as np
import torch

from rtdefects.segmentation import BaseSegmenter

logger = logging.getLogger(__name__)

# Storage for the model
_model: Optional[torch.nn.Module] = None
_model_path = Path(__file__).parent.joinpath('files', 'voids_segmentation_091321.pth')
# TODO (wardlt): Get the newer version of the model from Zhi-Gang


class PyTorchSegmenter(BaseSegmenter):
    """Implementing using Zhi-Ghang's Sept21 PyTorch Model"""

    def __init__(self, encoder: str = 'se_resnext50_32x4d'):
        # Get the preprocessor and build the preprocessing pipeline
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder)

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        self.preprocess = albu.Compose(_transform)

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        # Convert to RGB
        image = color.gray2rgb(image_data)

        # Perform the preprocessing
        image = self.preprocess(image=image)

        return image['image']

    def _load_model(self, device: str):
        global _model
        if _model is None:
            # Make sure the model exists
            if not _model_path.is_file():
                raise ValueError(f'Cannot find the model. No such file: {_model_path}')

            # Get the model hash to help with reproducibility
            with open(_model_path, 'rb') as fp:
                hsh = md5()
                while len(line := fp.read(4096 * 1024)) > 0:
                    hsh.update(line)
            logger.info(f'Loading the model from {_model_path}. MD5 Hash: {hsh.hexdigest()}')

            # Load it with Keras
            _model = torch.load(str(_model_path), map_location=device)
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
