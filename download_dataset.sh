#!/usr/bin/env bash

cd /ssd_scratch

mkdir datasets
cd datasets
# Download describable textures dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

mkdir mvtec
cd mvtec
# Download MVTec anomaly detection dataset
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz

# /ssd_scratch/datasets/mvtec/

# /ssd_scratch/Multi-class_MVTec-AD/MVTec-AD_20000step_bs32_eps_anomaly2_multiclass_0/checkpoint-20000/

# /ssd_scratch/2880f2ca379f41b0226444936bb7a6766a227587