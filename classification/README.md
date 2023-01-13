# Classification Stage & Evaluation

This directory contains an implementation of the classification stage and evaluation code.


## Environment Setup
We have provided a docker build script for each directory. Run the command below to setup docker container for training and evaluation environment:

```
docker build -t classification classification/docker/ # build docker image
docker run --gpus all -it -v path_to_ReCasNet_git/:/work/ --shm-size=20g classification #start docker container
cd classification/
```
## Data preparation

1. Download the dataset from  https://figshare.com/s/6c1ecad0107abc5bede3
2. Create the directory data/dataset/labelled-pool/CTC/
3. Move raw_cell_images/train and raw_cell_images/val from the dataset to data/dataset/labelled-pool/CTC/

After these steps, the file structure should look like this

```
classification
|---data
|    |--dataset
|    |  |--labelled-pool
|    |      |-CTC
|    |         |-train/
|    |             |-brightfield/
|    |             |-fluorescence/
|    |         |-val/
|    |             |-brightfield/
|    |             |-fluorescence/
|    |-...
|---...
```


## Training

Training and inference scripts of other models and datasets could be found in `script/`.

### Classification Stage Training
`-i`, and `-o` `-f` is config path, output path and dataset fraction (default is 1), respectively. The training dataset could be changed by modifying the config file.

```
python3 tools/train/train_CTC_torch.py -i config/CTC_brightfield_RGU.py -o CTC/Br_3cls
```

The model checkpoint and training log will be saved at `log/CTC/Br_3cls/`.

Similarly to the detection stage, the config `CTC_brightfield_RGU.py` could also be replaced to `CTC_brightfield_b_RGU.py` and `CTC_brightfield_fl_RGU.py`  to reproduce the result in Brightfield + Hoechst and Brightfield + Fluorescence setting, respectively.

## Evaluation

### Cell-level Evalutaion
The cell-level evalutaion of the pipeline could be evaluated using the following command:

```
python3 tools/test/eval.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls
```

### Image-level evaluation
The image-level evalutaion of the detection stage could be evaluated using the following command:

```
#format the output from the detection stage
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield.pkl \
-op CTC_result/CTC_brightfield_formatted.pkl

python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_formatted.pkl
```

`-ip` and `-op` is input and output pickle path, respectively.


To evaluate the performance of the whole pipeline, `python3 tools/test/CTC_inference_torch.py` also has to be called to perform an inference on the proposed objects along with the `--generate_image` flag in the `tools/utils/format_output.py`. The `-c` flag in `tools/test/CTC_inference_torch.py` indicates the confidence weight between the detection and classification stage (default is 1).

```
python3 tools/utils/format_output.py -ip ../detection/mmdetection/CTC_result/CTC_brightfield.pkl \
-op CTC_result/CTC_brightfield_formatted.pkl --generate_image

python3 tools/test/CTC_inference_torch.py -i config/CTC_brightfield_RGU.py -o CTC/publish/Br_3cls \
-ip CTC_result/CTC_brightfield_formatted.pkl -op CTC_result/CTC_brightfield_0_clsweight=1.pkl -c 1

python3 tools/test/eval_image_level.py -ip CTC_result/CTC_brightfield_clsweight=1.pkl
```