"""Test the segmentation function"""
from pathlib import Path

import numpy as np
from pytest import fixture
from skimage import io
import cv2

from rtdefects.function import perform_segmentation, analyze_defects


@fixture()
def image() -> np.ndarray:
    img = cv2.imread(str(Path(__file__).parent.joinpath('test-image.tif')))
    return np.array(img[None, :, :, :], np.float32) / 255


@fixture()
def mask() -> np.ndarray:
    img = io.imread(str(Path(__file__).parent.joinpath('test-image-mask.tif')))
    return np.array(img)


def test_run(image):
    output = perform_segmentation(image)
    assert output.shape == (1, 1024, 1024, 1)
    cv2.imwrite('test-image-mask.tif', output[0])


def test_analyze(mask):
    mask = mask > 0.99
    output = analyze_defects(mask)
    assert output['void_frac'] > 0
