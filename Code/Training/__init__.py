import torch

_device = torch.device("cuda:0")
# _device = torch.device("cpu")


def dev():
    return _device


def set_gpu(gpu_num):
    global _device
    _device = torch.device("cuda:" + str(gpu_num))
