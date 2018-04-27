# SINGA Docker Images

## Availabe images


| Tag | OS version | devel/runtime | Device|CUDA/CUDNN|Python|
|:----|:-----------|:--------------|:------|:---------|:-----|
|runtime| Ubuntu16.04|runtime|CPU|-|3.6|
|conda-cuda9.0| Ubuntu16.04|devel|GPU|CUDA9.0+CUDNN7.1.2|3.6|
|cuda9.0-py2| Ubuntu16.04|devel|GPU|CUDA9.0+CUDNN7.1.2|2.7|
|cuda9.0-py3| Ubuntu16.04|devel|GPU|CUDA9.0+CUDNN7.1.2|3.6|

runtime and conda-xxx image has installed miniconda3;
cudaxxx images have installed all depedent libs using apt-get.

## Usage

    docker pull nusdbsystem/singa:<Tag>
    docker run -it nusdbsystem/singa:<Tag> /bin/bash
    nvidia-docker run -it nusdbsystem/singa:<Tag> /bin/bash




