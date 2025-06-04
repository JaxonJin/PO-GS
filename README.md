<h1 align="center"><strong>PO-GS</strong></h1>
<h2 align="center">Perception-Oriented Structured 3D Gaussians for High-Fidelity View-Dependent Rendering</h2>
<p align="center">
<p align="center">
    
  </p>
  <h3 align="center"><a href="https://po-gs.github.io/pogs.github.io/">🌐 Project Page</a></h3>
  <div align="center"></div>
</p>

## Installation

```
conda create -n po_gs python=3.8
conda activate po_gs
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset

Mip-NeRF 360 dataset download link: [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)

Tanks&Temples and Deep Blending datasets download link: [Tanks&Temples and Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

## Data

The current directory should contain the following folders:

```
PO-GS
├───data/
    ├── dataset_name
    │   ├── scene1/
    │   │   ├── images
    │   │   │   ├── IMG_0.jpg
    │   │   │   ├── IMG_1.jpg
    │   │   │   ├── ...
    │   │   ├── sparse/
    │   │       └──0/
    │   ├── scene2/
    │   │   ├── images
    │   │   │   ├── IMG_0.jpg
    │   │   │   ├── IMG_1.jpg
    │   │   │   ├── ...
    │   │   ├── sparse/
    │   │       └──0/
    ...
```

## Training 

Similar to Scaffold-GS, we provide batch training scripts:

```
# Mip-NeRF 360
bash train_mip360.sh
# Tanks and Temples
bash train_tnt.sh
# Deep Blending
bash train_db.sh
```

## Evaluation 

For evalution, you can use the following command:

```
python render.py -m <path to trained model> 
python metrics.py -m <path to trained model> 
```



