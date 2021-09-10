# Implementation for **DTEP**

## Introduction
This repo provides codes of Unifying Debiased Domain Adaptation via Uncertainty Variance Modelling.

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
python main.py /data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --phase phase
```