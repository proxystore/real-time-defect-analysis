import numpy as np
from pytest import fixture

from rtdefects.analysis import track_voids, compute_drift


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
        np.array([[0, 0], [1.2, 1], [3, 3], [4, 4]]),  # All except the disappearing one
        np.array([[0, 0], [1.3, 1], [3, 3], [4, 4]]),  # All except the disappearing one
    ]


def test_tracking(example_tracks):
    tracks = track_voids(example_tracks, threshold=0.2)
    assert tracks.dtype == np.int
    assert (tracks == [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [2, 2, -1, -1],
        [-1, 3, 2, 2],
        [-1, -1, 3, 3]
    ]).all()

    # Compute the drifts
    drifts = compute_drift(tracks, example_tracks)
    assert (drifts == 0).all()

    # Apply a perturbation and see if detects it
    drifted_tracks = [np.add(x, [0, i]) for i, x in enumerate(example_tracks)]
    drifts = compute_drift(tracks, drifted_tracks)
    assert (drifts[:, 0] == 0).all()
    assert (drifts[:, 1] == [0, 1, 2, 3]).all()
