# !/usr/bin/python
# coding:utf-8
import os
import torch

JAVA_PATH = "C:\\Users\\38674\\.jdks\\openjdk-22"
CPU_CORE = min(os.cpu_count() - 1, 1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 1007

os.environ["JAVA_HOME"] = JAVA_PATH
