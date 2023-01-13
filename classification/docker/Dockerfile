FROM nvcr.io/nvidia/pytorch:22.07-py3


RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends -y libsm6 libxext6 libxrender-dev openslide-tools libgl1-mesa-glx
RUN pip install cython
RUN pip install matplotlib sklearn pandas  opencv-contrib-python==4.5.5.64 efficientnet albumentations pascal_voc_writer
WORKDIR /work

