import time
import torch
import torch.nn.functional as F

def featurize_batch(args, batch):
    N = batch.num_nodes
    edge_index = batch.edge_index.to(args.device)

    # generate random vector input
    x_in = torch.randn((N, args.rank), device=args.device)
    x_in = F.normalize(x_in, dim=1)

    # run model
    # TODO later, a more robust edge weight system
    num_edges = edge_index.shape[1]
    edge_weights = torch.ones(num_edges, device=args.device)
    return x_in, edge_index, edge_weights 


def train(args, model, train_loader, optimizer, criterion):
    '''Main training loop:

    Trains a model with an optimizer for a number of epochs
    '''
    epochs = args.epochs
    model_folder = args.log_dir

    model.to(args.device)
    for ep in range(epochs):
        start_time = time.time()
        total_obj = 0.
        for batch in train_loader:
            x_in, edge_index, edge_weights = featurize_batch(args, batch)
            x_out = model(x_in, edge_index, edge_weights)

            # get objective
            obj = criterion(x_out, edge_index)

            optimizer.zero_grad()
            obj.backward()
            optimizer.step()

            total_obj += obj.cpu().detach().numpy()

        epoch_time = time.time() - start_time
        print(f"epoch {ep} t={epoch_time} total_obj={total_obj:0.2f}")

        # TODO occasionally run validation and print loss
        if args.valid_epochs != 0 and ep % args.valid_epochs == 0:
            pass

        if args.save_epochs != 0 and ep % args.save_epochs == 0:
            torch.save(model.state_dict(), f"{model_folder}/model_ep{ep}.pt")

    # save trained model
    # TODO save best model, not just a bunch of epochs.
    torch.save(model.state_dict(), f"{model_folder}/model_ep{epochs}.pt")

def predict(model, loader, args):
    batches = []
    # TODO decide return signature and transform.
    for batch in loader:
        x_in, edge_index, edge_weights = featurize_batch(args, batch)
        x_out = model(x_in, edge_index, edge_weights)
        batches.append((x_out, edge_index))
    return batches

# these three functions
# plus possibly wrangling model output
