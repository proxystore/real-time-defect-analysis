"""Functions for the flask server"""
from pathlib import Path
from queue import Queue
from io import BytesIO

import imageio
import pandas as pd
from flask import Flask, send_file

_path = Path(__file__).parent
app = Flask('rtdefects', template_folder=str(_path / "templates"),
            static_folder=str(_path / "static"))


def _load_data() -> pd.DataFrame:
    """Load in the damage data from disk"""
    data_path = app.config['watch_dir'] / "masks" / "defect-details.json"
    if not data_path.is_file():
        return pd.DataFrame()
    return pd.read_json(data_path, lines=True)


@app.route('/')
def home():
    return send_file(_path / "templates" / "home.html")


@app.route('/latest-img.tif')
def latest_img():
    rad_data = _load_data()
    return send_file('test')


@app.route('/api/data')
def data():
    rad_data = _load_data()
    return rad_data.to_json(orient='records')


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
