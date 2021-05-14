from pathlib import Path
from hashlib import md5
import logging

import tensorflow as tf
import numpy as np

# Global variables for holding the model in memory
_model: tf.keras.models.Model = None
_model_path = Path(__file__).parent.joinpath('segment-model.h5')

logger = logging.getLogger(__name__)


def perform_segmentation(image_data: np.ndarray) -> np.ndarray:
    """Perform the image segmentation with a tensorflow model.

    Args:
        image_data: Image to be segmented, as a NumPy array
    Returns:
        Image segmentation mask
    """
    global _model  # Stores a loaded model between runs

    # If needed, load the model
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
        _model = tf.keras.models.load_model(_model_path)
        logger.info('Model loaded.')

    # Perform the segmentation
    return _model.predict(image_data)
