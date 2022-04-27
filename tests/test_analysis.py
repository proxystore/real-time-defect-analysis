import numpy as np
from pytest import fixture

from rtdefects.analysis import track_voids


@fixture
def example_tracks():
    """A few prototypical tracks over 3 frames:
    - Stay all frames (in the same point)
    - Move at a large rate
    - Disappear
    - Appear in new frame
    """

    # Make the frames
    return [
        np.array([[0, 0], [1, 1], [2, 2]]),  # All except the appearing particle,
        np.array([[1.1, 1], [0, 0], [2, 2], [3, 3]]),  # All, with two switching positions
        np.array([[0, 0], [1.2, 1], [3, 3]]),  # All except the disappearing one
        np.array([[0, 0], [1.2, 1], [3, 3]]),  # All except the disappearing one
    ]


def test_tracking(example_tracks):
    tracks = track_voids(example_tracks, threshold=0.2)
    assert (tracks == [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [2, 2, None, None],
        [None, 3, 2, 2]
    ]).all()
