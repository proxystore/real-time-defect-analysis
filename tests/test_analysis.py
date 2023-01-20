import numpy as np
import pandas as pd
import trackpy as tp
from pytest import fixture

from rtdefects.analysis import convert_to_per_particle, compile_void_tracks, compute_drift


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
        np.array([[0, 0], [1.3, 1], [3, 3], [4, 4], [5, 5]]),  # All except the disappearing one, add a new singlet
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
    assert len(particles) == 16


def test_tracking(example_segmentation):
    # Convert then run the analysis
    tracks = pd.concat(tp.link_df_iter(convert_to_per_particle(example_segmentation), search_range=0.2))
    assert len(tracks) == 16  # One per particle
    assert tracks['particle'].max() == 5  # 6 total particles

    # Gather the information
    track_ids = [g['local_id'].tolist() for _, g in tracks.groupby('particle')]
    assert (track_ids == [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [2, 2],
        [3, 2, 2],
        [3, 3],
        [4]
    ])

    # Compute the drifts
    drifts = compute_drift(tracks)
    assert np.isclose(drifts, 0).all()

    # Compile the tracks
    compile_void_tracks(tracks)
