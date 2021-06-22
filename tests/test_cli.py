from pathlib import Path
import json

from rtdefects.cli import _funcx_func


def test_funcx():
    data = Path(__file__).parent.joinpath("test-image.tif").read_bytes()
    mask, defect_info = _funcx_func(data)
    assert 0 < mask.mean() < 1,  "Mask is a single color."
    print(json.dumps(defect_info))
