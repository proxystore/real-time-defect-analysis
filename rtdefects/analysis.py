"""Functions to analyze segmented images"""
import logging
from typing import List, Optional, Dict

import pandas as pd
from scipy.stats import siegelslopes
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
        cur_centers = np.array(cur_centers)  # Make sure it is a numpy array
        last_centers = np.array(void_centers[cur_frame])  # Previous frame, noting that we are starting this loop from frame #1

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
            track = [-1] * (cur_frame + 1) + [u]
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
    for fid in range(1, len(positions)):
        # Get the voids in both images
        in_both, = np.where((tracks[:, fid - 1:fid + 1] >= 0).all(axis=1))

        # If there are no voids in both frames, assign a drift change of 0
        if sum(in_both) == 0:
            drifts.append(drifts[-1])
            continue

        # Get the median displacements displacements
        last_id = tracks[in_both, fid - 1]
        cur_id = tracks[in_both, fid]
        last_pos = np.array(positions[fid - 1])[last_id, :]
        cur_pos = np.array(positions[fid])[cur_id, :]
        median_disp = np.median(cur_pos - last_pos, axis=0)

        # Add the drift to that of the previous image
        drift = np.add(drifts[-1], median_disp)
        drifts.append(drift)

    return np.array(drifts)


def compile_void_tracks(void_positions: List[np.ndarray], tracks: np.ndarray,
                        void_radii: Optional[List[np.ndarray]] = None) -> pd.DataFrame:
    """Compile the void positions and tracks into a single array

    Also computes some key statistics of he

    Args:
        void_positions: Positions of voids in each frame
        tracks: Index of the same void across multiple frames
        void_radii: Radii of the voids in each frame
    Returns:
        Dataframe of the summary of voids
        - "start_frame": First frame in which the void appears
        - "end_frame": Last frame in which the void appears
        - "total_frames": Total number of frames in which the void appears
        - "positions": Positions of the void in each frame
        - "disp_from_start": How far the void has moved from the first frame
        - "max_disp": Maximum distance the void moved
        - "drift_rate": Average displacement from center over time
        - "dist_traveled": Total path distance the void has traveled
        - "total_travel": How far the void traveled over its whole life
        - "movement_rate": How far the void moves per frame
        - "radii": Radius of the void in each frame
        - "max_radius": Maximum radius of the void
        - "min_radius": Minimum radius of the void
        - "growth_rate": Median rate of change of the radius
    """

    # Loop over all unique voids
    voids = []
    for t, track in enumerate(tracks):
        # Get the frames where this void is visible
        visible_frames, = np.where(track >= 0)

        # Compute the displacement over each step
        positions = [void_positions[i][f] for i, f in enumerate(track) if f >= 0]
        positions = np.array(positions)

        # Gather some basic information about the void
        void_info = {
            'start_frame': np.min(visible_frames),
            'end_frame': np.max(visible_frames),
            'total_frames': np.sum(track >= 0),
            'positions': positions,
        }

        # If there is only one frame, we cannot do the following steps
        if positions.shape[0] > 1:
            # Compute the displacement from the start
            void_info['disp_from_start'] = np.linalg.norm(positions - positions[0, :], axis=1)
            void_info['max_disp'] = np.max(void_info['disp_from_start'])
            void_info['drift_rate'] = void_info['max_disp'] / void_info['total_frames']

            # Get the displacement for each step
            void_info['dist_traveled'] = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
            void_info['total_traveled'] = void_info['dist_traveled'][-1]
            void_info['movement_rate'] = void_info['total_traveled'] / void_info['total_frames']

        # More stats if we have radii
        if void_radii is not None:
            radii = [void_radii[i][f] for i, f in enumerate(track) if f >= 0]

            # Store some summary information
            void_info['radii'] = radii
            void_info['max_radius'] = max(radii)
            void_info['min_radius'] = min(radii)
            if len(radii) > 3:
                void_info['growth_rate'] = siegelslopes(radii)[0]

        # Add it to list
        voids.append(void_info)
    return pd.DataFrame(voids)
