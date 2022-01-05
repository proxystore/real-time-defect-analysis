"""Test the segmentation function"""
from pathlib import Path

from pytest import fixture, mark
import numpy as np
import imageio

from rtdefects.analysis import analyze_defects
from rtdefects.segmentation.tf import TFSegmenter


@fixture()
def image() -> np.ndarray:
    return imageio.imread(Path(__file__).parent.joinpath('test-image.tif'))


@fixture()
def mask() -> np.ndarray:
    img = imageio.imread(str(Path(__file__).parent.joinpath('test-image-mask.tif')))
    return np.array(img)


@mark.parametrize(
    'segmenter',
    [TFSegmenter()]
)
def test_run(image, segmenter):
    image = segmenter.transform_standard_image(image)
    output = segmenter.perform_segmentation(image)
    assert output.shape == (1, 1024, 1024, 1)
    imageio.imwrite('test-image-mask.tif', output[0])


def test_analyze(mask):
    mask = mask > 0.99
    output = analyze_defects(mask)
    assert output['void_frac'] > 0
