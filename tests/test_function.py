"""Test the segmentation function"""
from pathlib import Path

import numpy as np
from pytest import fixture
from skimage.io import imread

from rtdefects.function import perform_segmentation


@fixture()
def image() -> np.ndarray:
    img = imread(str(Path(__file__).parent.joinpath('test-image.png')))
    return img[None, :128, :128, :]


def test_run(image):
    perform_segmentation(image)
