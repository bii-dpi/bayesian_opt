import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_all_ligands
from models import GCN

from skopt.space import Real
from skopt.utils import use_named_args


# The list of hyper-parameters we want to optimize. For each one we define the
# bounds, the corresponding scikit-learn parameter name, as well as how to
# sample values from that dimension (`'log-uniform'` for the learning rate)
space  = [Real(10**-5, 10**0, "log-uniform", name='lr'),
          Real(10**-5, 10**0, "log-uniform", name='weight_decay')]

@use_named_args(space)
def get_validation_loss(no_cuda=True,
                        fastmode=True,
                        seed=42,
                        epochs=50,
                        lr=0.01,
                        weight_decay=5e-4,
                        hidden_dim_gcn=32,
                        output_dim=16,
                        dropout_prop=0.5,
                        hidden_dim_fc=32,
                        verbose=False
                        ):
    # Training settings
#                        help='Weight decay (L2 loss on parameters).')
    cuda = not no_cuda and torch.cuda.is_available()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    
    # Load data
    (training_features,
    validation_features,
    training_adj,
    validation_adj,
    training_labels,
    validation_labels) = load_all_ligands()
    
    
    # Model and optimizer
    model = GCN(init_dim=11,
                hidden_dim_gcn=hidden_dim_gcn,
                output_dim=output_dim,
                dropout_prop=dropout_prop,
                hidden_dim_fc=hidden_dim_fc)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    loss = nn.BCELoss()
    
    if cuda:
        model.cuda()
        training_features = [features.cuda() for features in training_features]
        validation_features = [features.cuda() for features in validation_features]
        training_adj = [adj.cuda() for adj in training_adj]
        validation_adj = [adj.cuda() for adj in validation_adj]
        training_adj = [adj.cuda() for adj in training_adj]
        validation_adj = [adj.cuda() for adj in validation_adj]
        training_labels = [labels.cuda() for labels in training_labels]
        validation_labels = [labels.cuda() for labels in validation_labels]
    
    def train(epoch):
        model.train()
        for i in range(len(training_labels)):
            optimizer.zero_grad()
            output = model(training_features[i], training_adj[i])
            loss_train = loss(output, training_labels[i])
            #acc_train = accuracy(output, labels)
            if i % 32 == 0:
                loss_train.backward()
                optimizer.step()
        if verbose:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()))
    
    
    def validate():
        model.eval()
        for i in range(len(validation_labels)):
            output = model(validation_features[i], validation_adj[i])
            loss_validation = loss(output, validation_labels[i])
            #acc_validation = accuracy(output, labels)
        if verbose:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_validation: {:.4f}'.format(loss_validation.item()))
        return loss_validation.item()
    
    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)
        if not fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            validate()
    
    if verbose:
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing
    return validate()
