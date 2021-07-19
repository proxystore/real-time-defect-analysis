# Demo Materials

This directory contains the scripts needed to demo the Real-time Defects labeling system.

1. Ensure the FuncX endpoint your system will use for compute is running
1. Delete and remake the frames directory: `rm -r frames; mkdir frames`
1. Launch the file-system watcher and web service: `rtdefects start frames`
1. Open a web browser to http://127.0.0.1:5000/
1. In a separate terminal, use `unpack_video.py` to generate a new frames to be labelled: `python update_video.py video_raw.avi`
