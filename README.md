# CSCH
Official Pytorch implementation of "Central Similarity Consistency Hashing for Asymmetric Image Retrieval"

## Overall training procedure of CSCH

<p align="center"><img src="figures/framework.png" width="900"></p>

## Train CSCH models
### Prepare datasets
We use public benchmark datasets: CIFAR-10, ImageNet, MS COCO. 
Image file name and corresponding labels are provided in ```./data```.

Datasets can be downloaded here:
<a href="https://github.com/swuxyj/DeepHash-pytorch">ImageNet and MS-COCO</a>

Example
- Train DHD model with ImageNet, AlexNet backbone, 64-bit, temperature scaling with 0.2
- ```python main_DHD.py --dataset=imagenet --encoder=AlexNet --N_bits=64 --temp=0.2``` 

```python main_DHD.py --help``` will provide detailed explanation of each argument.
