# Hierarchical B-Frame Video Coding Using Two-Layer CANF without Motion Coding (TLZMC)
This repository contains codes of a novel two-layer system hierarchical B-frame coding architecture without motion coding based on two-layer Conditional Augmented Normalization Flows (CANF) for video compression. Unlike traditional compression systems, our approach does not transmit any motion information, which explores a new direction for learned video coding. The motion coding is replaced using low-resolution learning-based compressor and merging operations. 

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Alexandre_Hierarchical_B-Frame_Video_Coding_Using_Two-Layer_CANF_Without_Motion_Coding_CVPR_2023_paper.pdf)]
[[Supplementary Material](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Alexandre_Hierarchical_B-Frame_Video_CVPR_2023_supplemental.pdf)]

More detail: https://nycu-clab.github.io

- [x] Evaluation
  - [x] Codes
  - [x] Checkpoints
  - [x] Requirements

## How to run evaluation
1. Install torchac using `python setup.py build` and `python setup.py install`
2. Install modules from requirements.txt
3. Run `python evaluate.py dataset_dir model_name checkpoint_name --group_gop n --gop m`

Example: 

- HEVC-B: `python evaluate.py ./dataset/class_b tlzmc-plus ./tlzmc-plus-mse-2048.ckpt --group_gop 3 --gop 32`

- UVG: `python evaluate.py ./dataset/class_b tlzmc-plus ./tlzmc-plus-mse-2048.ckpt --group_gop 18 --gop 32`

- Evaluation results are stored in folder `./evaluation`

There are three file examples to show demonstrate two-layer system:
- [x] TLZMC+ (DS2 (MaxPool2D), SR-CARN, Frame Synthesis) `model_name : tlzmc-plus`
- [x] TLZMC** (DS2 (MaxPool2D), SR-CARN, Multi-Frame Merging Network) `model_name: tlzmc-double-star`
- [x] TLZMC* (DS4 (MaxPool2D), SR-Net, Multi-Frame Merging Network) `model_name: tlzmc-star`

## Checkpoints
#### [TLZMC+](https://drive.google.com/drive/folders/1kIFgJCZisD4wLCDNLDBoWRcEUp7BC8Dc?usp=sharing) (DS2 (MaxPool2D), SR-CARN, Frame Synthesis, FTA) 

#### [TLZMC**](https://drive.google.com/drive/folders/1s3Ef8PQcR7ets8r5vm7D6JOaCqlzZglh?usp=sharing) (DS2 (MaxPool2D), SR-CARN, Multi-Frame Merging Network, FTA) 

#### [TLZMC*](https://drive.google.com/drive/folders/1X7B6wOTUnIOfoLz0Ao_gII50UhwQh19i?usp=sharing) (DS4, SR-Net, Multi-Frame Merging Network, FTA) 

Notes:
- The checkpoints are not final and can be subject to further fine-tuning.
- The CANF network are updated (less model size and computational complexity with comparable performance)

## Evaluation Dataset
#### HEVC-B
[Download](https://drive.google.com/drive/folders/10qv2TeJo9I2pBewzYm57IQ3OaYFjMVru?usp=sharing)

#### UVG (Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide)
[Download](https://ultravideo.fi/#testsequences)

## Citation

```
@InProceedings{Alexandre_2023_CVPR,
    author    = {Alexandre, David and Hang, Hsueh-Ming and Peng, Wen-Hsiao},
    title     = {Hierarchical B-Frame Video Coding Using Two-Layer CANF Without Motion Coding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10249-10258}
}
```

