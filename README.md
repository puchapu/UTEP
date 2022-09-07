# Implementation for **UTEP**

## Introduction
This repo provides codes of [Learning Unbiased Transferability for Domain Adaptation via Uncertainty Modeling](https://arxiv.org/pdf/2206.01319.pdf).

**This implementation is based on [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)**


## Prerequisites
```
pip3 install -r requirements.txt
```
## Train
Run the corresponding commands in run.sh for different tasks.
For example, run the following for Amazon to Webcam in Office-31
```
python main.py /data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
```
>Benefit from the [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library), If you run these tasks for the first time, the datasets will be downloaded to the corresponding path automatically

## Test or Analyse
With the phase setted to test or analysis, the code will run in different mode. Specifically, run the following for testing.
```
python main.py /data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase test
```
And run this for analysis 
```
python main.py /data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase analysis
```
If you are interested in our work and want to cite it, you can cite it as
```
@article{hu2022learning,
  title={Learning Unbiased Transferability for Domain Adaptation by Uncertainty Modeling},
  author={Hu, Jian and Zhong, Haowen and Yan, Junchi and Gong, Shaogang and Wu, Guile and Yang, Fei},
  journal={arXiv preprint arXiv:2206.01319},
  year={2022}
}
```
