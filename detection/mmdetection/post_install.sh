pip install -e .
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu114/torch1.13.0/index.html
pip uninstall opencv-contrib
pip install opencv-contrib-python==4.5.5.62
# git clone https://github.com/open-mmlab/mmcv.git
# cd mmcv
# MMCV_WITH_OPS=1 pip install -e .