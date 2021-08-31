# Detailed Installation Instructions

This document contains a detailed set of instructions for installing and performing a first set of runs for real-time defect analysis.

## Installing Git and Python Environment

We use git to perform version control for the software and Anaconda to keep track of the software dependencies.
It is possible to install the defect analysis code without either but, for the purposes of simplicity, we will not describe how to do so in this document.

- *Anaconda*: [Anaconda](https://www.anaconda.com/) is a tool for installing Python and other software tools in separate "environments" to simplify developing software with different requirements on the same computer. We recommend installing the command line version of Anaconda, "miniconda," which is [available for most operating systems](https://docs.conda.io/en/latest/miniconda.html). You can also install it without administrator privileges.
- *Installing Git*: [Git](https://git-scm.com/) is a popular system for tracking changes to software and coordinating development between many people. You can install git as a [standalone software](https://git-scm.com/downloads), via anaconda (i.e., call `conda install git` from the command line after installing miniconda), or as part of various GUI applications (e.g., [GitHub Desktop](https://desktop.github.com/)).

## Downloading and Installing `real-time-defect-analysis` 

The latest version of `real-time-defect-analysis` is available on our [GitHub page](https://github.com/globus-labs/real-time-defect-analysis).
Get the latest version using git to "clone" a local copy of the GitHub repository on your computer: `git clone https://github.com/globus-labs/real-time-defect-analysis.git`.
The cloning process will download the source code to your system to a new folder (named `real-time-defect-analysis`) that contains the configuration files needed to keep the folder up to date with GitHub. 
In the future, you can get the latest copy of the software by calling `git pull` from inside the `real-time-defect-analysis` folder.

The next step is to install a Python environment using Anaconda. 
The `environment.yml` file tells which software for Anaconda to install and is currently configured to install *everything* we need to run the package, including if you want to run the machine learning model locally. 
Build the environment by calling `conda env create --file environment.yml --force` from the command line.
Anaconda will then create a new environment, named "rtdefects," which you must activate before working with your tools.

## Authenticating with FuncX

Our application delegates the compute-intensive part of the image analysis to other computing systems using [FuncX](https://funcx.org/).
By default, our application is configured to run using a endpoint (a server that runs work for you) at ALCF and to use that resource you must:
1. *Get permission to use our ALCF endpoint.* We define a list of users using a Globus Group. Use [this link](https://app.globus.org/groups/37a33b10-00f2-11ec-9696-473e14106d31/about) to request access to the team.
2. *Install credentials on your computer.* You do not have to do anything now, but be prepared for our application to ask you to authenticate witreh FuncX the first time you run it.

## Launching the Application

Our application is started using a command line tool named `rtdefects`. 
Once you activate the Anaconda environment, you can get the most up-to-date documentation for how to launch it by calling `rtdefects start -h`. 
The output should look something like

```shell
(rtdefects) lward@bettik-linux:~$ rtdefects start -h
usage: rtdefects start [-h] [--regex REGEX] watch_dir

positional arguments:
  watch_dir      Which directory to watch for new files

optional arguments:
  -h, --help     show this help message and exit
  --regex REGEX  Regex to match files
```

As illustrated by the documentation: the key parameter for launching the application is the name of a directory to watch for micrographs. Launch the application by invoking it with the path to the directory (e.g., `rtdefects start output`).

Before running a real workload, we recommend you to test the application with example data provided with the source code.
First create an empty folder and start the application to watch that folder.

You may be asked to log in with Globus if this is your first time running `rtdefects` with a message like:
```
Please paste the following URL in a browser:
https://auth.globus.org/v2/oauth2/authorize?client_id=4cf29807-cf21-49ec-9443-ff9a3fb9f81c[...]
Please Paste your Auth Code Below: 
```
Go to the link given by the program then, as directed, paste the code provided by the system into the terminal. 

Once authenticated, you will see a few more authentication logging messages that likely end with one containing a URL for the web interface to your system.
```
2021-08-31 14:42:46,347 - werkzeug - INFO -  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
You can monoitor the progress of the software using this webpage and by following the output on your command line.

Test the system by copying the [test-image.tif](https://github.com/globus-labs/real-time-defect-analysis/blob/main/tests/test-image.tif) file provided with the software into your watch folder. A few things will then happen that you can watch in the logs.

1. *Reading the image from disk.* Make sure your system found the correct file
```2021-08-31 14:45:17,452 - rtdefects.cli - INFO - Read a 1.0 MB image from doc-test/test-image.tif```
2. *Submitting the processing request to FuncX*. The image will be uploaded to the FuncX web service along with instructions to run the image analysis at ALCF.
```2021-08-31 14:45:22,772 - rtdefects.cli - INFO - Submitted task to FuncX. Task ID: 8f4251a9-9709-4c71-89f5-f8e30589d4d3```
3. *Running the task at ALCF*. `rtdefects` will then wait for the image processing to complete at ALCF. The first inference request will take a few minutes for nodes to be requested and prepared to run our code. If all goes well you will see the system report waiting for a task then reporting a successful completion:
```
2021-08-31 14:45:22,772 - rtdefects.cli - INFO - Waiting for task request 8f4251a9-9709-4c71-89f5-f8e30589d4d3 to complete
2021-08-31 14:49:08,094 - rtdefects.cli - INFO - Result received for 1/1. RTT: 231.64s. Backlog: 0
2021-08-31 14:49:08,106 - rtdefects.cli - INFO - Wrote output file to: doc-test/masks/test-image.tif
```

Once you are finished testing, stop the program by <kbd>Ctrl</kbd> + <kbd>C</kbd>. FuncX will automatically release the resources it held for your experiment.
