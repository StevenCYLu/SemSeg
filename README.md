# Improving Semi-Supervised Semantic Segmentation with Sliced-Wasserstein Feature Alignment and Uniformity

Official implementation of **SWSeg**, a semi-supervised semantic segmentation method that uses Wasserstein distance and feature matching for learning from limited labeled data, accepted at CVPR 2025.

> **[Improving Semi-Supervised Semantic Segmentation
with Sliced-Wasserstein Feature Alignment and Uniformity](https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_Improving_Semi-Supervised_Semantic_Segmentation_with_Sliced-Wasserstein_Feature_Alignment_and_Uniformity_CVPR_2025_paper.pdf)**</br>
> Chen-Yi Lu, Kasra Derakhshandeh, Somali Chaterji</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2025*

## Requirements

### System Requirements
- Python 3.7+
- CUDA-capable GPU(s)
- Linux environment (recommended)

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### ADE20K Dataset

1. Download the ADE20K dataset from [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/)

2. Extract and organize the dataset:
```
/path/to/ade20k/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

3. Update the dataset path in `configs/ade20k.yaml`:
```yaml
data_root: "/path/to/your/ade20k/dataset"
```

## Training

### Quick Start

Use the provided training script for ADE20K:

```bash
# Basic usage (2 GPUs, 1/8 labeled data)
bash train_swseg_ade20k.sh

# Custom configuration
bash train_swseg_ade20k.sh [num_gpus] [master_port] [data_split]

# Example: 4 GPUs, port 12345, 1/4 labeled data
bash train_swseg_ade20k.sh 4 12345 1_4
```

### Manual Training Command

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=12345 \
    swseg.py \
    --config configs/ade20k.yaml \
    --labeled-id-path splits/ade20k/1_8/labeled.txt \
    --unlabeled-id-path splits/ade20k/1_8/unlabeled.txt \
    --save-path exp/ade20k/canopus/canopus_ade20k/1_8
```



## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{lu2025improving,
  title={Improving Semi-Supervised Semantic Segmentation with Sliced-Wasserstein Feature Alignment and Uniformity},
  author={Lu, Chen-Yi and Derakhshandeh, Kasra and Chaterji, Somali},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20233--20243},
  year={2025}
}
```

## Acknowledgments

- Code base is adapted from [UniMatch](https://github.com/LiheYoung/UniMatch)