# Revisiting the Critical Factors of  Augmentation-Invariant Representation Learning
--------
## Introduction
This repository is an official pytorch implementation of the paper [Revisiting the Critical Factors of  Augmentation-Invariant Representation Learning](https://arxiv.org/abs/2208.00275) (ECCV2022).

We release our framework to the public for the good of reproducibility. We hope it would be helpful for the community to develop new frameworks based on the fair benchmarking settings.

![architectures](https://github.com/megvii-research/revisitAIRL/blob/main/figures/arch.png)

## Main Results

|            | ImageNet(Acc) | URL  |
| ---------- | ------------- | ---- |
| MoCo v2    | 67.2          | [ckpt](https://drive.google.com/file/d/1F2_pJ50t6-INPM_OF3CMH10PBp7Jyt0y/view?usp=sharing) |
| MoCo v2+   | 72.4          | [ckpt](https://drive.google.com/file/d/1LDlE-3rSmO1sd1f8XfxQxFOxbO4j7ARG/view?usp=sharing) |
| S-MoCo v2+ | 71.2          | [ckpt](https://drive.google.com/file/d/13DFBKfFuuT61EyzLcRYCtXXjqgdQTfSo/view?usp=sharing) |
| BYOL SGD   | 72.1          | [ckpt](https://drive.google.com/file/d/1-5-O49vsro9YW9WTokSc8CoSrmjKfieB/view?usp=sharing) |


Notation:
1. The models are trained on ResNet-50 for 200 epochs.
2. We release the models with the best linear evaluation results, and the detailed hyper-parameters are packed into the checkpoint files.

## Usage
### Installation
First, clone the repository locally:
```
git clone https://github.com/megvii-research/revisitAIRL.git
```
Second, install the dependencies:
```
pip install -r requirements.txt
```

### Pre-training
For example, to train S-MoCov2+ on a single node with 8 GPUs:
```
python train.py -f exp/moco_based/mcv2p_symmlp.py --output-dir path/to/save -expn S-MoCov2p --data-path path/to/imagenet --total-epochs 200 --batch-size 256
```
Then, the process will create a directory ```path/to/save/S-MoCov2p``` to save all logs and checkpoints.

If you want to train a supervised model, please add the tag ```--single-aug```.

### Linear Evaluation
To evaluate a model trained with SGD optimizer on a single node with 8 GPUs:
```
python linear_eval.py --output-dir path/to/save -expn S-MoCov2p --data-path path/to/imagenet
```
Then, the process will create a directory ```path/to/save/S-MoCov2p/linear_eval``` to save all logs and checkpoints.

If you want to evaluate a model trained with LARS optimizer, please add the tag ```--large-norm-config```.


### Detection and segmentation
First, please install **detectron2** following [detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).


Second, convert the pre-trained model into a pickle format:
```
python convert-pretrain-to-detectron2.py path/to/pretrain_ckpt.pth.tar path/to/save/pickle_file.pkl [online/target/supervised]	
```
Then, a pickle file will be generated at ```path/to/save/pickle_file.pkl```.

For example, to transfer the pre-trained model to the VOC07 object detection task on a single node with 8 GPUs: 
```
python det/train_net.py  --config-file det/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 8 MODEL.WEIGHTS path/to/pickle_file.pkl OUTPUT_DIR path/to/save/voc07
```
Then, the process will create a directory ```path/to/save/voc07``` to save all logs and checkpoints.

### NormRescale
We provide a simple demo in ```norm_rescale.py```, please modify and use it as you like.

## Acknowledgements
Part of the code is adapted from previous works:

[MoCo](https://github.com/facebookresearch/moco)

We thank all the authors for their awesome repos.

## Citation
If you find this project useful for your research, please consider citing the paper.
```
@misc{huang2022revisiting,
      title={Revisiting the Critical Factors of Augmentation-Invariant Representation Learning}, 
      author={Junqiang Huang and Xiangwen Kong and Xiangyu Zhang},
      year={2022},
      eprint={2208.00275},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
If you have any questions, feel free to open an issue or contact us at [kongxiangwen@megvii.com](mailto:kongxiangwen@megvii.com).
