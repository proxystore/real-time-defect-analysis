from abc import ABCMeta, abstractmethod
from datetime import datetime
from threading import Thread
from argparse import ArgumentParser
from typing import Optional, List, Union, Iterator, Tuple
from pathlib import Path
from queue import Queue
from time import sleep
import logging
import json
import re

from funcx import FuncXClient
from ratelimit import sleep_and_retry, rate_limited
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent

from rtdefects.flask import app
from rtdefects.io import read_then_encode
from rtdefects.segmentation import BaseSegmenter
from rtdefects.segmentation.pytorch import PyTorchSegmenter
from rtdefects.segmentation.tf import TFSegmenter

logger = logging.getLogger(__name__)
_config_path = Path(__file__).parent.joinpath('config.json')


def _funcx_func(segmenter, data: bytes):
    """Function used for FuncX deployment of inference

    Inputs:
        segmenter (BaseSegmenter): Description of a segmentation tool
        data: TIF image data as a bytestring. Should be an 8-bit grayscale image
    Returns:
        - Bytes from a TIFF-encoded image of the mask
        - A dictionary of image analysis results
    """
    # Imports must be local to the function
    from rtdefects.analysis import analyze_defects
    from rtdefects.io import encode_as_tiff
    from io import BytesIO
    from time import perf_counter
    import numpy as np
    import imageio

    # Measure when we start
    start_time = perf_counter()

    # Load the TIFF file into a numpy array
    image_gray = imageio.imread(BytesIO(data))

    # Preprocess the image data
    image = segmenter.transform_standard_image(image_gray)

    # Perform the segmentation
    segment = segmenter.perform_segmentation(image)

    # Make it into a bool array
    segment = np.squeeze(segment)
    mask = segment > 0.9

    # Generate the analysis results
    defect_results = analyze_defects(mask)

    # Convert mask to a TIFF-encoded image
    message = encode_as_tiff(mask)

    # Add the execution time to the defect results
    defect_results['run_time'] = perf_counter() - start_time
    return message, defect_results


def _set_config(function_id: Optional[str] = None, endpoint_id: Optional[str] = None):
    """Set the system configuration given the parser"""

    # Read in the current configuration
    if _config_path.is_file():
        logger.info(f'Loading previous settings from {_config_path}')
        with open(_config_path, 'r') as fp:
            config = json.load(fp)
    else:
        config = {}

    # Define the configuration settings
    if function_id is not None:
        config['function_id'] = function_id
    if endpoint_id is not None:
        config['endpoint_id'] = endpoint_id

    # Print out the configuration settings
    for key, value in config.items():
        logger.info(f'New setting for {key}: {value}')

    # Save it
    with open(_config_path, 'w') as fp:
        json.dump(config, fp, indent=2)
    logger.info(f'Wrote configuration to {_config_path}')


def _register_function():
    """Register the inference function with FuncX"""

    client = FuncXClient()

    # Get the Group UUID
    config = json.loads(_config_path.read_text())
    function_id = client.register_function(_funcx_func, group=config['group_uuid'])
    _set_config(function_id=function_id)


class ImageProcessEventHandler(FileSystemEventHandler, metaclass=ABCMeta):
    """Base class for providers which perform image analysis asynchronously.

    Implementations must provide two methods:
        `submit_file` which reads the file and sends it to self._queue to be executed, updates self.index
            to indicate a file was submitted for execution
        `iterate_results` which iterates over completed inference records, removing objects from queue when completed
    """

    def __init__(self, segmenter: BaseSegmenter, file_regex: Optional[str] = None):
        """
        Args:
            segmenter: Description of the segmentation tool
            file_regex: Regex string to match file formats
        """
        self.segmenter = segmenter
        self._queue = Queue()
        self.index = 0
        self.file_regex = re.compile(file_regex) if file_regex is not None else None

    @property
    def queue(self):
        """Queue used to store in-progress results"""
        return self._queue

    @abstractmethod
    def submit_file(self, file_path: Path, detect_time: datetime):
        """Submit a file to be analyzed

        Args:
            file_path: Path to the file to be analyzed
            detect_time: Time at which a file was detected
        """
        pass

    @abstractmethod
    def iterate_results(self) -> Iterator[Tuple[Path, bytes, dict, float, datetime]]:
        """Iterate over results from process images

        Yields:
            - Path to the original image
            - Segmented image as a byte string
            - Information about the detected defects
            - Time between file detection and result received, seconds
            - Time the image was first detected
        """

    def on_created(self, event: Union[FileCreatedEvent, DirCreatedEvent]):
        # Ignore directories
        if event.is_directory:
            logger.info('Created object is a directory. Skipping')
            return

        # Send the file to be analyzed
        detect_time = datetime.now()
        file_path = Path(event.src_path)

        # Match the filename
        if self.file_regex is not None:
            if self.file_regex.match(file_path.name.lower()) is None:
                logger.info(f'Filename "{file_path}" did not match regex. Skipping')

        # Wait for write to finish
        sleep(1.)  # TODO (wardlt): Implement a more intelligent way to check for write finish

        # Attempt to run it
        try:
            self.submit_file(file_path, detect_time)
        except BaseException as e:
            logger.warning(f'Submission for {event.src_path} failed. Error: {e}')


class FuncXSubmitEventHandler(ImageProcessEventHandler):
    """Submit an image processing task to FuncX when an image file is created"""

    def __init__(self, segmenter: BaseSegmenter, client: FuncXClient, func_id: str, endp_id: str,
                 file_regex: Optional[str] = None):
        """
        Args:
            segmenter: Description of segmentation tool
            client: FuncX client
            func_id: ID of the image processing function
            endp_id: Endpoint ID for execution
            file_regex: Regex string to match file formats
        """
        super().__init__(segmenter, file_regex)
        self.client = client
        self.func_id = func_id
        self.endp_id = endp_id

    def submit_file(self, file_path: Path, detect_time: datetime):
        # Load the image from disk
        image_data = read_then_encode(file_path)
        logger.info(f'Read a {len(image_data) / 1024 ** 2:.1f} MB image from {file_path}')

        # Submit it to FuncX for evaluation
        task_id = self.client.run(self.segmenter, image_data, function_id=self.func_id, endpoint_id=self.endp_id)
        logger.info(f'Submitted task to FuncX. Task ID: {task_id}')

        # Push the task ID and submit time to the queue for processing
        self.index += 1
        self.queue.put((task_id, detect_time, Path(file_path)))

    def iterate_results(self) -> Iterator[Tuple[Path, bytes, dict, float, datetime]]:
        # Set up a throttling for the FuncX request
        @sleep_and_retry
        @rate_limited(self.client.max_requests - 10, period=self.client.period * 0.9)
        def throttled_call(task_id):
            return self.client.get_task(task_id)

        for task_id, detect_time, img_path in iter(self.queue.get, None):
            logger.info(f'Waiting for task request {task_id} to complete')

            # Wait it for it finish from FuncX
            while (task := throttled_call(task_id))['pending']:
                continue
            mask, defect_info = self.client.get_result(task_id)
            if mask is None:
                logger.warning(f'Task failure: {task["exception"]}')
                break
            rtt = (datetime.now() - detect_time).total_seconds()

            yield img_path, mask, defect_info, rtt, detect_time


class LocalProcessingHandler(ImageProcessEventHandler):
    """Image handler that performs image analysis locally"""

    def submit_file(self, file_path, detect_time):
        # Push the task ID and submit time to the queue for processing
        self.index += 1
        self.queue.put((detect_time, Path(file_path)))

    def iterate_results(self) -> Iterator[Tuple[Path, bytes, dict, float, datetime]]:
        for detect_time, img_path in iter(self.queue.get, None):
            # Load the image from disk
            image_data = read_then_encode(img_path)
            logger.info(f'Read a {len(image_data) / 1024 ** 2:.1f} MB image from {img_path}')

            # Run the function
            mask, defect_info = _funcx_func(self.segmenter, image_data)
            rtt = (datetime.now() - detect_time).total_seconds()

            yield img_path, mask, defect_info, rtt, detect_time


def main(args: Optional[List[str]] = None):
    """Launch service that automatically processes images and displays results as a web service"""

    # Make the argument parser
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Which mode to launch the server in', required=True)

    # Add in the configuration settings
    config_parser = subparsers.add_parser('config', help='Define the configuration for the server')
    config_parser.add_argument('--function-id', help='UUID of the function to be run')
    config_parser.add_argument('--funcx-endpoint', help='FuncX endpoint on which to run image processing')

    # Add in the launch setting
    start_parser = subparsers.add_parser('start', help='Launch the processing service')
    start_parser.add_argument('--model', choices=['tf', 'pytorch'], default='pytorch',
                              help='Which segmentation model to use')
    start_parser.add_argument('--regex', default=r'.*.tiff?$', help='Regex to match files')
    start_parser.add_argument('--redo-existing', action='store_true', help='Submit any existing files in the directory')
    start_parser.add_argument('--local', action='store_true', help='Perform image analysis locally,'
                                                                   ' instead of via FuncX')
    start_parser.add_argument('watch_dir', help='Which directory to watch for new files')

    # Add in the register setting
    subparsers.add_parser('register', help='(Re)-register the funcX function')

    # Parse the input arguments
    args = parser.parse_args(args)

    # Make the logger
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Handle the configuration
    if args.command == 'config':
        return _set_config(function_id=args.function_id, endpoint_id=args.funcx_endpoint)
    elif args.command == 'register':
        return _register_function()

    assert args.command == 'start', f'Internal Error: The command "{args.command}" is not yet supported. Contact Logan'

    # Select the correct segmenter
    if args.model == 'tf':
        segmenter = TFSegmenter()
    elif args.model == 'pytorch':
        segmenter = PyTorchSegmenter()
    else:
        raise ValueError(f'Model type "{args.model}" is not supported yet')

    # Prepare the event handler
    if args.local:
        handler = LocalProcessingHandler(segmenter=segmenter, file_regex=args.regex)
    else:
        client = FuncXClient()
        client.max_request_size = 50 * 1024 ** 2
        with open(_config_path, 'r') as fp:
            config = json.load(fp)
        handler = FuncXSubmitEventHandler(segmenter, client, config['function_id'],
                                          config['endpoint_id'], file_regex=args.regex)

    # Prepare the watch directory
    watch_dir = Path(args.watch_dir)
    mask_dir = watch_dir.joinpath('masks')
    mask_dir.mkdir(exist_ok=True)

    # Launch the flask app
    app.config['exec_queue'] = handler.queue
    app.config['watch_dir'] = Path(args.watch_dir)
    flask_thr = Thread(target=app.run, daemon=True, name='rtdefects.flask')
    flask_thr.start()

    # Launch the watcher
    obs = Observer()
    obs.schedule(handler, path=args.watch_dir, recursive=False)
    obs.start()

    # If desired, submit the existing files
    data_path = mask_dir.joinpath('defect-details.json')
    if args.redo_existing:
        data_path.unlink(missing_ok=True)  # Delete any existing data
        for file in watch_dir.iterdir():
            if file.is_file():
                handler.submit_file(file, datetime.now())

    # Wait for results to complete
    try:
        for index, (img_path, mask, defect_info, rtt, detect_time) in enumerate(handler.iterate_results()):
            # Report the completed result
            logger.info(f'Result received for {index + 1}/{handler.index}. RTT: {rtt:.2f}s.'
                        f' Backlog: {handler.queue.qsize()}')

            # Save the mask to disk
            out_name = mask_dir.joinpath(img_path.name)
            with out_name.open('wb') as fp:
                fp.write(mask)
            logger.info(f'Wrote output file to: {out_name}')

            # Write out the image defect information
            defect_info['created_time'] = datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
            defect_info['completed_time'] = datetime.now().isoformat()
            defect_info['mask-path'] = str(out_name)
            defect_info['image-path'] = str(img_path)
            defect_info['rtt'] = rtt
            defect_info['detect_time'] = detect_time.isoformat()
            with data_path.open('a') as fp:
                print(json.dumps(defect_info), file=fp)
    except KeyboardInterrupt:
        logger.info('Detected an interrupt. Stopping system')
    except BaseException:
        obs.stop()
        logger.warning('Unexpected failure!')
        raise

    # Shut down the file reader
    obs.stop()
    obs.join()
