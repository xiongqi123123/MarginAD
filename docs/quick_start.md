# Quick Start

### Set up a new virtual environment
```bash
conda create -n MarginAD python=3.8 -y
conda activate MarginAD
```

### Install dependency packpages
```bash
MarginAD_path="path/to/MarginAD"
cd ${MarginAD_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```
:fire: Tips
1. It is necessary to install the `mmcv-full` version of `mmcv`. It is recommended to use `cuda113` and `torch1.11.0`, and to install `mmcv` version `1.7.2`. Use the following command: 
   
   ```
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. Remember to install `flash-attn` offline. The download address for `flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl` is [here](https://github.com/Dao-AILab/flash-attention/releases). Move the file to the `MomAD` folder and execute the following command to install:

   ```
   pip install flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl
   ```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${MarginAD_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```


### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing

####  train MarginAD
```
bash ./tools/dist_train.sh \
   projects/configs/Margin_Stage2_Full.py \
   8 \
```
