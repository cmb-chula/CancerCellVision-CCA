# Detection Stage

Implementation of the detection stage.

## Environment Setup
We have provided a docker build script for each directory. Run the command below to setup the docker container for the training and evaluation environment:

```
docker build -t detection detection/mmdetection/docker/ # build docker image
docker run --gpus all -it -v path_to_ReCasNet_git/:/work/ --shm-size=20g detection #start docker container
cd detection/mmdetection/
sh post_install.sh
```

## Data Preparation

1. Download the dataset from  https://figshare.com/s/6c1ecad0107abc5bede3
2. Create the directory data/dataset/CTC
3. Move raw_images_for_model/brightfield, raw_images_for_model/fluorescence from the dataset to data/dataset/CTC
4. Move Annotations/trainval, Annotations/test, Annotations/train.txt, Annotations/val.txt, Annotations/test.txt from the dataset to data/dataset/CTC
5. Rename trainval directory to Annotations/

After these steps, the file structure should look like this

```
mmdetection
|---data
|    |--dataset
|        |-CTC
|          |-brightfield/
|          |-fluorescence/
|          |-Annotations/
|          |-test/
|          |-train.txt
|          |-val.txt
|          |-test.txt
|---...
```


## Training

Training and inference scripts of other models and datasets could be found in `script/`.

### Detection Stage Training

The code below shows how to train the detector under Brightfield setting:
```
python tools/train.py  configs/CTC/CTC_brightfield.py --seed 40 --work-dir checkpoints/CTC_brightfield_0
```

`CTC_brightfield.py` could be replaced to `CTC_brightfield_b.py` and `CTC_brightfield_fl.py` to reproduce the result in Brightfield + Hoechst and Brightfield + Fluorescence setting, respectively.

## Inference
The code below shows how to perform an inference from the trained detector on the test set under the Brightfield setting:
```
python tools/test.py  configs/CTC/CTC_brightfield.py checkpoints/CTC_brightfield_0/epoch_8.pth --out CTC_result/CTC_brightfield.pkl
```
The `--out` argument indicates the prediction output path. Similarly to the training command, the config `CTC_brightfield.py` could also be replaced to `CTC_brightfield_b.py` and `CTC_brightfield_fl.py`.
