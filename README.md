# Visual Grounding with Transformers


##  Overview

This repository includes PyTorch implementation and trained models of VGTR(**V**isual **G**rounding with **TR**ansformers).

[[arXiv](https://arxiv.org/abs/2105.04281)]

> Abstract
>>In this paper, we propose a transformer based approach for visual grounding. Unlike existing proposal-and-rank frameworks that rely heavily on pretrained object detectors or proposal-free frameworks that upgrade an off-the-shelf one-stage detector by fusing textual embeddings, our approach is built on top of a transformer encoder-decoder and is independent of any pretrained detectors or word embedding models. Termed as VGTR – Visual Grounding with TRansformers, our approach is designed to learn semantic-discriminative visual features under the guidance of the textual description without harming their location ability. This information flow enables our VGTR to have a strong capability in capturing context-level semantics of both vision and language modalities, rendering us to aggregate accurate visual clues implied by the description to locate the interested object instance. Experiments show that our method outperforms state-of-the-art proposal-free approaches by a considerable margin on four benchmarks.

<img width="805" alt="图片" src="https://user-images.githubusercontent.com/83934424/157177788-534d16e8-c91c-432d-8939-213c7f3065a2.png">


## Prerequisites

- python 3.6
- pytorch>=1.6.0
- torchvision
- CUDA>=9.0
- others (opencv-python etc.)


## Preparation

   1. Clone this repository.

   2. Data preparation.

      Download Flickr30K Entities from [Flickr30k Entities (bryanplummer.com)](http://bryanplummer.com/Flickr30kEntities/) and  [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) 

      Download MSCOCO images from [MSCOCO](http://images.cocodataset.org/zips/train2014.zip)

      Download processed indexes from [Gdrive](https://drive.google.com/drive/folders/1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ?usp=drive_open), process by [zyang-ur
   ](https://github.com/zyang-ur/onestage_grounding).

   3. Download backbone weights. We use resnet-50/101 as the basic visual encoder. The weights are pretrained on MSCOCO, and can be downloaded here (BaiduDrive):

      [ResNet-50](https://pan.baidu.com/s/1ZHR_Ew8tUZH7gZo1prJThQ)(code：ru8v);  [ResNet-101](https://pan.baidu.com/s/1zsQ67cUZQ88n43-nmEjgvA)(code：0hgu).

   4. Organize all files like this：

   ```bash
   .
   ├── main.py
   ├── store
   │   ├── data
   │   │   ├── flickr
   │   │   │   ├── corpus.pth
   │   │   │   └── flickr_train.pth
   │   │   ├── gref
   │   │   └── gref_umd
   │   ├── ln_data
   │   │   ├── Flickr30k
   │   │   │   └── flickr30k-images
   │   │   └── other
   │   │       └── images
   │   ├── pretrained
   │   │   └── flickr_R50.pth.tar
   │   └── pth
   │       └── resnet50_detr.pth
   └── work
   ```


## Model Zoo

   | Dataset           | Backbone  | Accuracy            | Pretrained Model (BaiduDrive)                                |
   | ----------------- | --------- | ------------------- | ------------------------------------------------------------ |
   | Flickr30K Entites | Resnet50  | 74.17               | [flickr_R50.pth.tar](https://pan.baidu.com/s/1VUnxD-5pXnM7iFwIl8q9kA) code: rpdr |
   | Flickr30K Entites | Resnet101 | 75.32               | [flickr_R101.pth.tar](https://pan.baidu.com/s/10GcUFLSTei9Lwvu4e5GjrQ) code: 1igb |
   | RefCOCO           | Resnet50  | 78.70  82.09  73.31 | [refcoco_R50.pth.tar](https://pan.baidu.com/s/1GIe5OoOQOADYc1vVGcSXbw) code: xjs8 |
   | RefCOCO           | Resnet101 | 79.30  82.16  74.38 | [refcoco_R101.pth.tar](https://pan.baidu.com/s/1GL-itH93G_e3VVNUPtocSA) code: bv0z |
   | RefCOCO+          | Resnet50  | 63.57  69.65  55.33 | [refcoco+_R50.pth.tar](https://pan.baidu.com/s/1PUF8WoTrOLmYU24kgAMXKQ) code: 521n |
   | RefCOCO+          | Resnet101 | 64.40  70.85  55.84 | [refcoco+_R101.pth.tar](https://pan.baidu.com/s/1mJiA7i7-Mp5ZL5D6dEDy0g) code: vzld |
   | RefCOCOg          | Resnet50  | 62.88               | [refcocog_R50.pth.tar](https://pan.baidu.com/s/1KvDPisgSLzy8u5bIVCBiOg) code: wb3x |
   | RefCOCOg          | Resnet101 | 64.05               | [refcocog_R101.pth.tar](https://pan.baidu.com/s/13ubLIbIUA3XlhzSOjaK7dg) code: 5ok2 |
   | RefCOCOg-umd      | Resnet50  | 65.62  65.30        | [umd_R50.pth.tar](https://pan.baidu.com/s/1-PgzbA98rUOl7VJHAO-Exw) code: 9lzr |
   | RefCOCOg-umd      | Resnet101 | 66.83  67.28        | [umd_R101.pth.tar](https://pan.baidu.com/s/1JkGbYL8Of3WOVWI9QcVwhQ) code: zen0 |


## Train

   ```
   python main.py \
      --gpu $gpu_id \
      --dataset $[refcoco | refcoco+ | else] \
      --batch_size $bs \
      --savename $exp_name \
      --backbone $[resnet50 | resnet101] \
      --cnn_path $resnet_coco_weight_path
   ```
   

## Inference

   Download the pretrained models and put it into the folder ```./store/pretrained/```.

   ```
   python main.py \
      --test \
      --gpu $gpu_id \
      --dataset $[refcoco | refcoco+ | else] \
      --batch_size $bs \
      --pretrain $pretrained_weight_path
   ```

## Acknowledgements

Part of codes are from:

   1. [facebookresearch/detr](https://github.com/facebookresearch/detr)；
   2. [zyang-ur/onestage_grounding](https://github.com/zyang-ur/onestage_grounding)； 
   3. [andfoy/refer](https://github.com/andfoy/refer)；
   4. [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).


 
## Citation
   ```
   @article{du2021visual,
     title={Visual grounding with transformers},
     author={Du, Ye and Fu, Zehua and Liu, Qingjie and Wang, Yunhong},
     journal={arXiv preprint arXiv:2105.04281},
     year={2021}
   }
   ```
   
