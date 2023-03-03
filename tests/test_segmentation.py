"""Test the segmentation function"""
from math import isclose
from pathlib import Path

from pytest import fixture, mark
import numpy as np
import imageio

from rtdefects.analysis import analyze_defects
from rtdefects.segmentation.pytorch import PyTorchSegmenter, download_model
from rtdefects.segmentation.tf import TFSegmenter


@fixture()
def image() -> np.ndarray:
    return imageio.imread(Path(__file__).parent.joinpath('test-image.tif'))


@fixture()
def mask() -> np.ndarray:
    img = imageio.imread(str(Path(__file__).parent.joinpath('test-image-mask.tif')))
    return np.array(img)


def test_download(tmpdir):
    from rtdefects.segmentation import pytorch
    orig = pytorch._model_dir
    try:
        pytorch._model_dir = tmpdir
        download_model('README.md')
        assert (tmpdir / 'README.md').read_text('ascii').startswith('#')
    finally:
        pytorch._model_dir = orig


@mark.parametrize(
    'segmenter',
    [TFSegmenter(), PyTorchSegmenter('voids_segmentation_091321.pth'), PyTorchSegmenter('voids_segmentation_030323.pth')]
)
def test_run(image, segmenter):
    image = segmenter.transform_standard_image(image)
    assert isinstance(image, np.ndarray)
    output = segmenter.perform_segmentation(image)
    output = np.squeeze(output)
    assert output.shape == (1024, 1024)
    imageio.imwrite('test-image-mask.tif', output)
    assert isclose(output.mean(), 0.03, abs_tol=0.01)


def test_analyze(mask):
    mask = mask > 0.99
    output = analyze_defects(mask)
    assert output['void_frac'] > 0
