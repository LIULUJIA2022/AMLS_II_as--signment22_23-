# AMLS_II_as--signment22_23-

[train/test]_images the image files.

roughly 15,000 images in the test set.

train.csv

image_id the image file name. 

label the ID code for the disease：

0

1

2

3

4


#import required libraries
import numpy as np

import glob

import cv2

import os

import pandas as pd

import matplotlib.pyplot as plt

import shutil

import paddle

import math

from paddle.static import InputSpec

from visualdl import LogWriter


# System environment 
GPU version: RTX4080

CUDA version: 10.2

GPU Compute Capability: 8.9 

Driver API Version: 12.0

Runtime API Version: 10.2

Windows 11

