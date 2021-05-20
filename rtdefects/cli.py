import re
from argparse import ArgumentParser
from typing import Optional, List, Union
from pathlib import Path
from queue import Queue
from time import perf_counter, sleep
import logging
import json

from funcx import FuncXClient
from skimage.io import imread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent


logger = logging.getLogger(__name__)
_config_path = Path(__file__).parent.joinpath('config.json')


def _funcx_func(data):
    from rtdefects.function import perform_segmentation
    return perform_segmentation(data)


def _set_config(function_id: Optional[str] = None, endpoint_id: Optional[str] = None):
    """Set the system configuration given the parser
    """

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

    # Save it
    with open(_config_path, 'w') as fp:
        json.dump(config, fp, indent=2)
    logger.info(f'Wrote configuration to {_config_path}')


def _register_function():
    """Register the inference function with FuncX"""

    client = FuncXClient()

    function_id = client.register_function(_funcx_func)
    _set_config(function_id=function_id)


class FuncXSubmitEventHandler(FileSystemEventHandler):
    """Submit a image processing task to FuncX when an image file is created"""

    def __init__(self, client: FuncXClient, func_id: str, endp_id: str, queue: Queue, file_regex: Optional[str] = None):
        """
        Args:
             client: FuncX client
             func_id: ID of the image processing function
             endp_id: Endpoint ID for execution
             queue: Queue to push results to
             file_regex: Regex string to match file formats
        """
        super().__init__()
        self.client = client
        self.func_id = func_id
        self.endp_id = endp_id
        self.queue = queue
        self.index = 0
        self.file_regex = re.compile(file_regex) if file_regex is not None else None

    def on_created(self, event: Union[FileCreatedEvent, DirCreatedEvent]):
        # Ignore directories
        if event.is_directory:
            logger.info('Created object is a directory. Skipping')
            return

        # Match the filename
        if self.file_regex is not None:
            file_path = Path(event.src_path)
            if self.file_regex.match(file_path.name) is None:
                logger.info(f'Filename "{file_path}" did not match regex. Skipping')

        # Performance information
        detect_time = perf_counter()
        self.index += 1

        # Load the image from disk
        image_data = imread(event.src_path)
        logger.info(f'Read a {image_data.shape[0]}x{image_data.shape[1]} image from {event.src_path}')

        # Submit it to FuncX for evaluation
        # TODO (wardlt): Shape the image to the proper size for our models
        task_id = self.client.run(image_data[None, :128, :128, :], function_id=self.func_id, endpoint_id=self.endp_id)
        logger.info(f'Submitted task to FuncX. Task ID: {task_id}')

        # Push the task ID and submit time to the queue for processing
        self.queue.put((task_id, detect_time, self.index))


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

    # Prepare the event handler
    client = FuncXClient()
    client.max_request_size = 50 * 1024 ** 2
    with open(_config_path, 'r') as fp:
        config = json.load(fp)
    exec_queue = Queue()
    handler = FuncXSubmitEventHandler(client, config['function_id'], config['endpoint_id'], exec_queue)

    # Prepare the watcher
    obs = Observer()
    obs.schedule(handler, path=args.watch_dir, recursive=False)
    obs.start()
    while True:
        try:
            # Wait for a task to be added the queue
            task_id, detect_time, index = exec_queue.get(timeout=3600)

            # Wait it for it finish from FuncX
            while (task := client.get_task(task_id))['pending']:
                sleep(1)
            result = task.pop('result')
            if result is None:
                logger.warning(f'Task failure: {task["exception"]}')
                break
            rtt = perf_counter() - detect_time
            logger.info(f'Result received for {index}/{handler.index}. Round-trip time: {rtt:.2f}s. Backlog: {exec_queue.qsize()}')
        except KeyboardInterrupt:
            logger.info('Detected an interrupt. Stopping system')
            break
        except BaseException:
            obs.stop()
            logger.warning('Unexpected failure!')
            raise
    obs.stop()
    obs.join()
