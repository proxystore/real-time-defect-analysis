import numpy as np
import pandas as pd
import trackpy as tp
from pytest import fixture
from trackpy import compute_drift

from rtdefects.analysis import track_voids, convert_to_per_particle


@fixture
def example_segmentation():
    """A few prototypical tracks over 3 frames:
    - Stay all frames (in the same point)
    - Move at a large rate
    - Disappear
    - Appear in new frame
    """

    # Make the frames
    tracks = [
        np.array([[0, 0], [1, 1], [2, 2]]),  # All except the appearing particle,
        np.array([[1.1, 1], [0, 0], [2, 2], [3, 3]]),  # All, with two switching positions
        np.array([[0, 0], [1.2, 1], [3, 3], [4, 4]]),  # All except the disappearing one
        np.array([[0, 0], [1.3, 1], [3, 3], [4, 4]]),  # All except the disappearing one
    ]
    frames = np.arange(4)
    radii = [np.ones((t.shape[0],)) for t in tracks]

    # Make the dataframe
    return pd.DataFrame({
        'positions': tracks,
        'frames': frames,
        'radii': radii
    })


def test_conversion(example_segmentation):
    """Test converting to a trackpy-ready format"""
    particles = pd.concat(list(convert_to_per_particle(example_segmentation)))
    assert len(particles) == 15


def test_tracking(example_segmentation):
    # Convert then run the analysis
    tracks = pd.concat(tp.link_df_iter(convert_to_per_particle(example_segmentation), search_range=0.2))
    assert len(tracks) == 15  # One per particle
    assert tracks['particle'].max() == 4  # 5 total particles

    # Gather the information
    track_ids = [g['local_id'].tolist() for _, g in tracks.groupby('particle')]
    assert (track_ids == [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [2, 2],
        [3, 2, 2],
        [3, 3]
    ])

    # Compute the drifts
    drifts = tp.compute_drift(tracks)
    assert (drifts['y'] == 0).all()
    assert (drifts['x'] > 0).all()
