"""Functions for the flask server"""
from pathlib import Path
from queue import Queue
from io import BytesIO
import json

import imageio
import numpy as np
import pandas as pd
from bokeh.embed import json_item
from bokeh.plotting import figure
from bokeh.models import AjaxDataSource
from skimage import color
from flask import Flask, send_file
from flask_cors import CORS

_path = Path(__file__).parent
app = Flask('rtdefects', template_folder=str(_path / "templates"),
            static_folder=str(_path / "static"))
CORS(app)


def _load_data() -> pd.DataFrame:
    """Load in the damage data from disk"""
    data_path = app.config['watch_dir'] / "masks" / "defect-details.json"
    if not data_path.is_file():
        return pd.DataFrame()
    data = pd.read_json(data_path, lines=True)
    data['detect_time'] -= data['detect_time'].min()
    return data


@app.route('/')
def home():
    return send_file(_path / "templates" / "home.html")


@app.route('/bokeh')
def full_plot():
    # Set up to connect with self to get data
    source = AjaxDataSource(data_url='http://localhost:5000/api/data',
                            polling_interval=5000)

    # Make
    fig = figure(height=400, sizing_mode='stretch_width')

    # Make a simple plot
    fig.line('detect_time', 'void_frac', source=source)

    fig.xaxis.axis_label = 'Time (s)'
    fig.yaxis.axis_label = 'Void Fraction'

    return json.dumps(json_item(fig, "plot"))


@app.route('/api/data', methods=('GET', 'POST'))
def get_data():
    rad_data = _load_data()
    return rad_data.to_dict(orient='list')


@app.route('/api/status')
def started():
    # Access the state of the system
    queue = app.config['exec_queue'] if 'exec_queue' in app.config else Queue()
    rad_data = _load_data()  # TODO (wardlt): Save this in memory too?

    # Get the data that is always available
    output = {
        'started': len(_load_data()) > 0,
        'num_evaluated': len(rad_data),
        'backlog': queue.qsize()
    }

    # Get stuff that requires the system to have run once
    if len(rad_data) > 0:
        output['last_image'] = rad_data['image-path'].iloc[-1]
    return output


@app.route('/api/image/<int:image>')
def get_image(image: int):
    """Get a certain image from the dataset

    Args:
        image: Index of the image
    Returns:
        Image in JPEG format
    """
    # Get the image
    rad_data = _load_data()
    image_path = Path(rad_data.iloc[image]['image-path'])

    # Convert it to JPEG
    img = imageio.imread(image_path)
    img_out = BytesIO()
    imageio.imwrite(img_out, img, format='jpeg')
    img_out.seek(0)
    return send_file(img_out, attachment_filename=image_path.name, mimetype='image/jpeg')


@app.route('/api/mask/<int:image>')
def get_mask(image: int):
    """Get a certain mask from the dataset

    Args:
        image: Index of the mask
    Returns:
        Image in JPEG format
    """
    # Get the image
    rad_data = _load_data()
    image_path = Path(rad_data.iloc[image]['mask-path'])

    # Convert it to JPEG
    img = imageio.imread(image_path)
    img_out = BytesIO()
    imageio.imwrite(img_out, img, format='jpeg')
    img_out.seek(0)
    return send_file(img_out, attachment_filename=image_path.name, mimetype='image/jpeg')


@app.route('/api/overlay/<int:image>')
def get_overlay(image: int):
    """Get a certain mask from the dataset

    Args:
        image: Index of the mask
    Returns:
        Image in JPEG format
    """
    # Get the image
    rad_data = _load_data()
    image_path = Path(rad_data.iloc[image]['image-path'])
    base = imageio.imread(image_path)
    mask_path = Path(rad_data.iloc[image]['mask-path'])
    mask = imageio.imread(mask_path)

    # Convert base to rgb
    base_rgb = np.array(color.gray2rgb(base), dtype=np.float)

    # Make the mask red
    mask_rgb = np.array(color.gray2rgb(mask), dtype=np.float)
    mask_rgb[:, :, 1:] = 0

    # Compute the output
    output = base_rgb + mask_rgb
    output = np.clip(output, 0, 255)
    output = np.array(output, dtype=np.uint8)

    # Convert it to JPEG
    img_out = BytesIO()
    imageio.imwrite(img_out, output, format='jpeg')
    img_out.seek(0)
    return send_file(img_out, attachment_filename=image_path.name, mimetype='image/jpeg')
