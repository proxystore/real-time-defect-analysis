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

The first step is to launch the FuncX endpoint that will handle the image processing requests.
**More notes TBD**

Then, launch the file system watcher via **TBD**.

## Support

This material is based upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357.
