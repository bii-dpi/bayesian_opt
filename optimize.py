
import numpy as np
from progressbar import progressbar
from train import get_validation_loss

from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize

space  = [Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Real(10**-5, 10**0, "log-uniform", name='weight_decay')]

res_gp = gp_minimize(get_validation_loss, space, n_calls=50, random_state=0)

print("Best score=%.4f" % res_gp.fun)

from skopt.plots import plot_convergence

plot_convergence(res_gp)