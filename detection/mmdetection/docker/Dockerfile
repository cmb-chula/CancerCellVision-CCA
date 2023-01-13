FROM nvcr.io/nvidia/pytorch:20.07-py3

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx openslide-tools\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MMCV_WITH_OPS=1


RUN pip install openslide-python
# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
# RUN pip install -U openmim
# RUN mim install mmcv-full==1.3.17


RUN pip install --no-cache-dir mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu114/torch1.6.0/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .
WORKDIR /work