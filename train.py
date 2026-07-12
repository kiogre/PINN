import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("="*90)

    torch.seed(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')