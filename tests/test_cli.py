from datetime import datetime
from pathlib import Path
from io import BytesIO
import json

from imageio import imread

from rtdefects.cli import _funcx_func, LocalProcessingHandler
from rtdefects.segmentation.pytorch import PyTorchSegmenter
from rtdefects.segmentation.tf import TFSegmenter

test_image = Path(__file__).parent.joinpath("test-image.tif")


def test_funcx():
    """Test the funcx function"""
    data = test_image.read_bytes()
    mask_bytes, defect_info = _funcx_func(TFSegmenter(), data)
    mask = imread(BytesIO(mask_bytes), format='tiff')
    assert 0 < mask.mean() < 255, "Mask is a single color."
    assert mask.max() == 255
    print(json.dumps(defect_info))


def test_local_reader(tmpdir):
    reader = LocalProcessingHandler(PyTorchSegmenter())
    reader.submit_file(test_image, datetime.now())
    img_path, mask, defect_info, rtt, detect_time = next(reader.iterate_results())
