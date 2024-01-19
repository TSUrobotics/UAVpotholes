# Pothole Segmentation and Area Estimation in Python
Ethan Welborn ([email](mailto:ethanwelborn@protonmail.com), [website](https://www.ethanwelborncs.com)) and Sotirios Diamantas ([email](mailto:diamantas@tarleton.edu), [website](https://sites.google.com/site/sotiriosresearch/))

Original paper title: "Pothole Segmentation and Area Estimation with Deep Neural Networks and Unmanned Aerial Vehicles"

[Link to paper](https://link.springer.com/chapter/10.1007/978-3-031-47966-3_29)

## How To Use
Download the docker image ``docker pull datamachines/cudnn_tensorflow_opencv:11.3.1_2.9.1_4.6.0-20220815``, and run it using the ``run.sh`` file as a template (you will have to change the folder mounting to match your system).

Inside the docker container, run the ``install.sh`` file to install the ``ultralytics`` and ``opencv-python-headless`` files.

Finally, change the configuration values in ``areaEstimation.py`` to your liking and run it using Python 3.
