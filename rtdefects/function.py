from pathlib import Path
from hashlib import md5
import logging

from skimage import data, filters, measure, morphology
import tensorflow as tf
import numpy as np

# Global variables for holding the model in memory
_model: tf.keras.models.Model = None
_model_path = Path(__file__).parent.joinpath('segment-model.h5')

logger = logging.getLogger(__name__)


def perform_segmentation(image_data: np.ndarray) -> np.ndarray:
    """Perform the image segmentation with a tensorflow model.

    Args:
        image_data: Images to be segmented. Should be 3 channels with values ranging between 0-1 as np.float32
    Returns:
        Image segmentation mask
    """
    global _model  # Stores a loaded model between runs

    # Check the shape
    assert image_data.ndim == 4, "Expects a stack of images"
    assert image_data.shape[-1] == 3, "Expects 16 output channels"
    assert image_data.dtype == np.float32, "Expects np.float32"
    assert 0 <= np.min(image_data) and np.max(image_data) <= 1, "Image values should be in [0, 1]"

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


def analyze_defects(mask: np.ndarray, min_size: int = 50) -> dict:
    """Analyze the voids in a masked image

    Args:
        mask: Masks for a defect image
        min_size: Minimum size of defects
    Returns:
        List of the computed properties
    """

    # Clean up the mask
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, min_size)
    mask = morphology.binary_erosion(mask, morphology.square(1))
    output = {'void_frac': mask.sum() / (mask.shape[0] * mask.shape[1])}

    # Assign labels to the labeled regions
    labels = measure.label(mask)
    output['void_count'] = labels.max()

    # Compute region properties
    props = measure.regionprops(labels, mask)
    radii = [p['equivalent_diameter'] for p in props]
    output['radii'] = radii
    output['radii_average'] = np.average(radii)

    return output
