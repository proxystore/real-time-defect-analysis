# Adding a New Model

You need to tell `rtdefects` how to use a model and then upload that model to ALCF's file storage.

1. Add the encoder associated with the model in the `_encoders` dictionary
   in the top of [pytorch.py](../rtdefects/segmentation/pytorch.py).
   They key of the dictionary is the name of the pth file and the value is the name of the encoder.
2. Upload the model to [ALCF](https://app.globus.org/file-manager?origin_id=f10a69a9-338c-4e5b-baa1-0dc92359ab47&origin_path=%2Fivem%2Fmodels%2F).
   We need the PTH file.