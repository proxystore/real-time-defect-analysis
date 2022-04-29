"""Functions to analyze segmented images"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd
from skimage import measure, morphology
import numpy as np

logger = logging.getLogger(__name__)


def analyze_defects(mask: np.ndarray, min_size: int = 50) -> dict:
    """Analyze the voids in a masked image

    Args:
        mask: Masks for a defect image
        min_size: Minimum size of defects
    Returns:
        List of the computed properties
    """

    # Clean up the mask
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, min_size)
    mask = morphology.binary_erosion(mask, morphology.square(1))
    output = {'void_frac': mask.sum() / (mask.shape[0] * mask.shape[1])}

    # Assign labels to the labeled regions
    labels = measure.label(mask)
    output['void_count'] = int(labels.max())

    # Compute region properties
    props = measure.regionprops(labels, mask)
    radii = [p['equivalent_diameter'] for p in props]
    output['radii'] = radii
    output['radii_average'] = np.average(radii)
    output['positions'] = [p['centroid'] for p in props]
    return output


def track_voids(void_centers: List[np.ndarray], threshold: float, lookbehind: int = 1) -> np.ndarray:
    """Determine the index of each void in each frame, if possible

    Args:
        void_centers: List of the centers of voids from each frame
        threshold: Allowed distance between particles
        lookbehind: How many frames to look back for a match
    Returns:
        The index of each unique void in each of the frames.
        Values -1 corresponds to the void being absent in a particular frames.
    """

    # Start a list of the tracks with the IDs of the points from the first frame
    #  The tracks will contain the ID of a void over each frame number
    tracks: List[List[Optional[int]]] = [[i] for i in range(len(void_centers[0]))]

    # Start with the first frame and move forward
    assert lookbehind == 1, "We haven't yet implemented further looking"
    last_map: Dict[int, int] = dict((i, i) for i in range(len(void_centers[0])))  # Map of ID from last frame to ID in `tracks`
    for cur_frame, cur_centers in enumerate(void_centers[1:]):
        # Get the points in the next frame
        last_centers = void_centers[cur_frame]  # Previous frame, noting that we are starting this loop from frame #1

        # Get the distances between voids in each frame
        dists = np.linalg.norm(cur_centers[:, None, :] - last_centers[None, :, :], axis=-1)  # Shape: <n current> x <n previous>
        valid_match = dists < threshold

        # Ensure we only match one void from the previous frame, the closest one
        is_closest = np.zeros_like(valid_match, dtype=bool)
        closest_ind = np.argmin(dists, axis=1)
        is_closest[np.arange(is_closest.shape[0]), closest_ind] = True
        valid_match = np.logical_and(valid_match, is_closest)

        # Ensure that each void from the previous frame only matches one in the new, the closest one
        is_closest = np.zeros_like(valid_match, dtype=bool)
        closest_ind = np.argmin(dists, axis=0)
        is_closest[closest_ind, np.arange(is_closest.shape[1])] = True
        valid_match = np.logical_and(valid_match, is_closest)

        # Find those points which are < threshold apart,
        #  the closest match for each particle in the new frame
        #  and the closest match for the particle in the old frame
        cur_frame_id, last_frame_id = np.where(valid_match)

        # Map each of the matched particles to a position in tracks
        new_map: Dict[int, int] = dict()
        for cid, lid in zip(cur_frame_id, last_frame_id):
            global_id = last_map[lid]
            tracks[global_id].append(cid)
            new_map[cid] = global_id

        # Add unmatched projectiles from the current frame to the list
        unmatched = set(range(len(cur_centers))).difference(cur_frame_id)
        for u in unmatched:
            global_id = len(tracks)
            track = [-1] * (cur_frame + 1) + [global_id]
            tracks.append(track)
            new_map[u] = global_id

        # Mark that tracks from the previous frame(s) are not found
        unmatched = set(range(len(tracks))).difference(new_map.values())
        for u in unmatched:
            tracks[u].append(-1)

        # Now that we're done, update the tracking map and move on
        last_map = new_map

    return np.array(tracks, dtype=np.int)


def compute_drift(tracks: np.ndarray, positions: List[np.ndarray]) -> np.ndarray:
    """Estimate the drift for each frame from the positions of voids that were mapped between multiple frames

    We determine the "drift" based on the median displacement of all voids, which is based
    on the assumption that there is no net motion of all the voids.

    Args:
        tracks: Index of specific voids across multiple image frames
        positions: Positions of each void in each frame
    Returns:
        Drift correction for each frame
    """

    # We'll assume that the first frame has a void
    drifts = [(0, 0)]

    # We're going to go frame-by-frame and guess the drift from the previous frame
    for fid in range(1, len(tracks)):
        # Get the voids in both images
        in_both, = np.where((tracks[:, fid - 1:fid + 1] >= 0).all(axis=1))

        # Get the median displacements displacements
        last_id = tracks[in_both, fid - 1]
        cur_id = tracks[in_both, fid]
        median_disp = np.median(positions[fid][cur_id, :] - positions[fid - 1][last_id, :], axis=0)

        # Add the drift to that of the previous image
        drift = np.add(drifts[-1], median_disp)
        drifts.append(drift)

    return np.array(drifts)
