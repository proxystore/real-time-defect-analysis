# Real-Time Defect Identification

This package is a service for real-time defect identification from TEM images. 

The key component is a server that will watch a specific folder on a file system and 
send images to be processed by a Tensorflow-based image process pipeline. 
The results of the labeling will be saved to disk, and the summaries will be made
available via a web service.

The server is intended to use FuncX to perform image processing on remote resources. 
This package also includes the environment needed to service image processing requests.

## Installation

The environment can be installed via Anaconda: ``conda env create --file environment.yml --force``

Change the tensorflow dependency in `environment.yml` to `tensorflow-gpu` if GPU support is desired.

## Use

Our system requires configuring both an endpoint to run the image analysis and one where the images are generated. 

### Compute Provider

The Compute Provider should be a system with the ability to process images quickly and need not have access
to the filesystem that stores the images. It only needs internet access. To set up the ability to run the functions:

1. Install and activate the computational environment, [as described above](#installation).
1. Configure a FuncX endpoint to receive image processing commands. Follow the directions on [the Funcx documentation](https://funcx.readthedocs.io/en/latest/endpoints.html#first-time-setup)
to configure the system.
    1. Record the ID produced by your endpoint, which should look something like `8c01d13c-cfc1-42d9-96d2-52c51784ea17`.  
1. Place the Keras `.h5` file with the model in the `rtdefects` directory and name it `segment-model.h5`
1. Launch the FuncX service (e.g., `funcx-endpoint start default`).

Once you are done processing images, you may stop the FuncX endpoint. 
You will need to stop and re-start each time you change the Keras model. 

### Image Analyzer

The Image Analyzer must have access to the images and an internet connection.

First, install and activate the computational environment, [as described above](#installation).

Before any image analysis runs, you must tell the system what computations to run and where to run them:
1. Register the analysis function by calling `rtdefects register`
1. Configure where to run them using the endpoint ID for the [Compute Provider](#compute-provider)
and setting it using `rtdefects config --funcx-endpoint <the endpoint id>`
   1. You may later change this endpoint using `rtdefects config`

Begin an analysis run by calling `rtdefects start`.
The single argument for this command is a directory to watch for new files,
though more will soon be implemented.

The process will watch for files to be created, send those files
to be processed via FuncX, and then collect the completed results.
It will run until you exit by pressing <kbd>Ctrl</kbd>+<kbd>C</kbd>,
which will require up to 15s to register on a Windows system.

## Support

This material is based upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357.
