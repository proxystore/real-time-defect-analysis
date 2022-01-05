"""Functions related to segmentation and analysis of microscopy images"""

import numpy as np


# TODO (wardlt): Support segmentation methods besides semantic segmentation
class BaseSegmenter:
    """Base class for implementations of a segmentation tool

    Implementations must provide a function for reshaping from the format we use
    to transmit images (unit8-based grayscale) into whatever is expected by this specific model,
    and a function that performs the segmentation and returns a boolean array mask.
    """

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        """Transform an image into a format compatible with the model

        Args:
            image_data: Image in the as-transmitted format: unit8 grayscale
        Returns:
            Image in whatever form needed by the model
        """
        raise NotImplementedError

    def perform_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Perform the image segmentation

        Args:
            image_data: Images to be segmented.
        Returns:
            Image segmentation mask as a boolean array
        """
        raise NotImplementedError
