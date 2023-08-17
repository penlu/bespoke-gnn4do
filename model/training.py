import time
import torch
import torch.nn.functional as F

def train(args, model, train_loader, optimizer, criterion):
    '''Main training loop:

    Trains a model with an optimizer for a number of epochs
    '''
    epochs = args.epochs
    model_folder = args.log_dir

    for ep in range(epochs):
        print('epoch: ', ep)
        for batch in train_loader:
            N = batch.num_nodes
            edge_index = batch.edge_index

            # generate random vector input
            x_in = torch.randn((N, args.rank), dtype=torch.float)
            x_in = F.normalize(x_in, dim=1)

            # run model
            # TODO more robust edge weight system
            num_edges = edge_index.shape[1]
            edge_weights = torch.ones(num_edges)
            x_out = model(x_in, edge_index, edge_weights)

            # get objective
            obj = criterion(x_out, edge_index)

            optimizer.zero_grad()
            obj.backward()
            optimizer.step()

            # TODO occasionally run validation and print loss
            #if ep % 2000 == 0:
            #    torch.save(conv_mc['lift'].state_dict(), f"{model_folder}/lift_ep{epochs}.pt")
            #    torch.save(conv_mc['project'].state_dict(), f"{model_folder}/project_ep{epochs}.pt")

def test():
    pass

def predict():
    pass

# these three functions
# plus possibly wrangling model output
