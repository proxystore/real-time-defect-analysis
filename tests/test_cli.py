from pathlib import Path
from io import BytesIO
import json

from imageio import imread

from rtdefects.cli import _funcx_func
from rtdefects.segmentation.tf import TFSegmenter


def test_funcx():
    data = Path(__file__).parent.joinpath("test-image.tif").read_bytes()
    mask_bytes, defect_info = _funcx_func(TFSegmenter(), data)
    mask = imread(BytesIO(mask_bytes), format='tiff')
    assert 0 < mask.mean() < 255,  "Mask is a single color."
    assert mask.max() == 255
    print(json.dumps(defect_info))
