"""Test the segmentation function"""
from pathlib import Path

import numpy as np
from pytest import fixture
import cv2

from rtdefects.function import perform_segmentation


@fixture()
def image() -> np.ndarray:
    img = cv2.imread(str(Path(__file__).parent.joinpath('test-image.tif')))
    return np.array(img[None, :, :, :], np.float32) / 255


def test_run(image):
    output = perform_segmentation(image)
    assert output.shape == (1, 1024, 1024, 1)
    cv2.imwrite('test-image-mask.tif', output[0])
