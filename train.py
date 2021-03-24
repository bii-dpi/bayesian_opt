import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_all_ligands
from models import GCN

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden-dim-gcn', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--output-dim', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout-prop', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--hidden-dim-fc', type=int, default=32,
                        help='Number of hidden units.')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    (training_features,
    validation_features,
    training_adj,
    validation_adj,
    training_labels,
    validation_labels) = load_all_ligands()
    
    
    # Model and optimizer
    model = GCN(init_dim=11,
                hidden_dim_gcn=args.hidden_dim_gcn,
                output_dim=args.output_dim,
                dropout_prop=args.dropout_prop,
                hidden_dim_fc=args.hidden_dim_fc)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.BCELoss()
    
    if args.cuda:
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
    
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()))
    
    
    def validate():
        model.eval()
        for i in range(len(validation_labels)):
            output = model(validation_features[i], validation_adj[i])
            loss_validation = loss(output, validation_labels[i])
            #acc_validation = accuracy(output, labels)
    
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_validation: {:.4f}'.format(loss_validation.item()))
        return loss_validation.item()
    
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            validate()
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing
    return validate()

if __name__ == "__main__":
    main()
