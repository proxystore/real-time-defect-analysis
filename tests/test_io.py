from pathlib import Path

import imageio
from pytest import mark
import numpy as np

from rtdefects.io import load_file, read_then_encode

_my_path = Path(__file__).parent


@mark.parametrize(
    'filename', ['test-image.tif', 'test-image.dm4']
)
def test_load(filename: str):
    file = _my_path.joinpath(filename)
    data = load_file(file)
    assert data.ndim == 2
    assert data.dtype == np.float32


def test_transmit():
    message = read_then_encode(_my_path.joinpath('test-image.tif'))
    new_data = imageio.imread(message, format='tiff')
    assert new_data.dtype == np.uint8
