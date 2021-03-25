
import numpy as np
from progressbar import progressbar
from train import get_validation_loss

for weight_decay in progressbar(np.linspace(0, 1, 10)):
    for lr in np.linspace(0, 1, 10):
        print(f"{weight_decay}, {lr}, {get_validation_loss(1, 10)}")
