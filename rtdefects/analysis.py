"""Functions to analyze segmented images"""
import logging

from skimage import measure, morphology
import numpy as np

logger = logging.getLogger(__name__)


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
    output['void_count'] = int(labels.max())

    # Compute region properties
    props = measure.regionprops(labels, mask)
    radii = [p['equivalent_diameter'] for p in props]
    output['radii'] = radii
    output['radii_average'] = np.average(radii)

    return output
