"""Test the segmentation function"""
from pathlib import Path

from pytest import fixture
from skimage import color
import numpy as np
import imageio

from rtdefects.function import perform_segmentation, analyze_defects


@fixture()
def image() -> np.ndarray:
    img_gray = imageio.imread(Path(__file__).parent.joinpath('test-image.tif'))
    img = color.gray2rgb(img_gray)
    return np.array(img[None, :, :, :], np.float32) / 255


@fixture()
def mask() -> np.ndarray:
    img = imageio.imread(str(Path(__file__).parent.joinpath('test-image-mask.tif')))
    return np.array(img)


def test_run(image):
    output = perform_segmentation(image)
    assert output.shape == (1, 1024, 1024, 1)
    imageio.imwrite('test-image-mask.tif', output[0])


def test_analyze(mask):
    mask = mask > 0.99
    output = analyze_defects(mask)
    assert output['void_frac'] > 0
