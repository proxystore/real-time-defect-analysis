from argparse import ArgumentParser
from pathlib import Path
from time import sleep
import logging

from skimage.color import rgb2gray
from skimage.transform import resize
from imageio.plugins.ffmpeg import FfmpegFormat
from imageio import get_reader, imwrite
import numpy as np

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Make an argument parser
parser = ArgumentParser()
parser.add_argument('--time-between-frames', '-t', default=15, help='Minimum time between writing frames', type=float)
parser.add_argument('--output-directory', '-o', default='frames', help='Directory in which to output frames')
parser.add_argument('--output-resolution', '-r', default=1024, help='Resolution of the output folder', type=int)
parser.add_argument('--maximum-frames', '-m', default=None, help='Maximum number of frames to write out', type=int)
parser.add_argument('video_file', help='Video file to unpack')

args = parser.parse_args()

# Open the video
reader: FfmpegFormat.Reader = get_reader(args.video_file)
frame_count = reader.count_frames()
to_write = frame_count if args.maximum_frames is None else args.maximum_frames
logging.info(f'Opened {args.video_file}. Frame count: {frame_count}')

# Get the output dir
out_dir = Path(args.output_directory)
out_dir.mkdir(exist_ok=True)

# Write frames out at specified resolutions
for i, frame in zip(range(to_write), reader.iter_data()):
    # Convert to grayscale and resize
    gray_image = rgb2gray(frame)
    resized_image = resize(gray_image, output_shape=(args.output_resolution,) * 2)
    ready_to_write = np.array(resized_image * 255, dtype=np.uint8)

    # Save it to disk
    out_file = out_dir / f'frame_{i}.tif'
    imwrite(out_file, ready_to_write)
    logging.info(f'Processed frame {i + 1}/{to_write}. Saved to disk as {out_file}')

    # Sleep for designated time
    sleep(args.time_between_frames)
