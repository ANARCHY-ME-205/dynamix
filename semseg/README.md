# SEMANTIC SEGMENTATION


## Live workings of the semantic segmentation model.

https://github.com/ANARCHY-ME-205/semseg/assets/129314735/22f779cf-a043-4b29-839f-3818c6688f16

https://github.com/ANARCHY-ME-205/semseg/assets/129314735/8bcab38b-7e29-457a-bbd4-2185c43b5570

## Prerequisites

# THANK, WORSHIP, PRAY to Tamoghna!!!

This requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

#### Installing pytorch

```shell
pip install torch;
pip install torchvision
```

### Installing dependencies : 

```shell
pip install testresources ;
pip install launchpadlib ;
pip install --upgrade pip setuptools ;
pip install --upgrade six 
```

## Installation

**Step 1.** Installing mmengine and mmcv using openmim

```shell
pip install -U openmim ;
mim install mmengine ;
mim install "mmcv>=2.0.0" 
```
**Step 2.** Install MMSegmentation.

```shell
cd semseg ; 
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### Verify the installation

To verify whether MMSegmentation is installed correctly, we provide some sample codes to run an inference demo.

**Step 0.** Checking mmseg version.

```python
import mmseg
print(mmseg.__version__)
# Example output: 1.0.0
```
### The following steps to verify installation are optional

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py` and `pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmsegmentation from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where segmentation masks are covered on all objects.

## Running semantic segmentation.

**Run semseg.py**

### Plausible semseg models : 

Option(A). **configs/bisenetv1/bisenetv1_r18-d32-in1k-pre_4xb4-160k_cityscapes-1024x1024.py  && bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210905_220251-8ba80eff.pth** :   
gives a decent computational speed of semseg although the accuracy is compromised a bit (70%).

Option(B). **configs/pspnet/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py && pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth** :      
this one is found to have the highest amount accuracy(81%) so far but very very slow computational speed something around 3 fps which is very bad.

Option(C). **configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py && ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth** :   
model link : https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth
works with moderate accuracy and moderate speed. I will be using this for now. DISCLAIMER : Need the bolt zed cam param tuning for this to work well!!!



