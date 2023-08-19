import time
import torch
import torch.nn.functional as F
from model.saving import save_model

def featurize_batch(args, batch):
    N = batch.num_nodes
    edge_index = batch.edge_index.to(args.device)

    # generate random vector input
    x_in = torch.randn((N, args.rank), dtype=torch.float, device=args.device)
    x_in = F.normalize(x_in, dim=1)

    # run model
    # TODO later, a more robust edge weight system
    num_edges = edge_index.shape[1]
    edge_weights = torch.ones(num_edges, device=args.device)
    return x_in, edge_index, edge_weights 

def validation(args, model, val_loader, baseline, projector, score):
    '''Run a brief validation.
    '''

    if args.baseline == 'sdp':
        baseline = None # TODO a function that runs sdp
    elif args.baseline == 'autograd':
        baseline = None # TODO a function that runs autograd

    if args.projector == 'e1':
        projector = None # TODO just take the first element
    elif args.projector == 'random_hyperplane':
        projector = None # TODO random hyperplane, or even best of N random hyperplanes

    if args.problem == 'max_cut':
        score = None # TODO max cut scoring function i.e. (E - obj) / 2.
    else:
        pass # TODO

    base_scores = []
    model_scores = []
    for batch in val_loader:
        x_in, edge_index, edge_weights = featurize_batch(args, batch)

        # run baseline and model
        x_base = baseline(x_in, edge_index, edge_weights)
        x_model = model(x_in, edge_index, edge_weights)

        # run projection method on both baseline and model output
        x_proj_base = projector(x_base, edge_index, edge_weights)
        x_proj_model = projector(x_model, edge_index, edge_weights)

        score_base = score(x_base, edge_index, edge_weights)
        score_model = score(x_model, edge_index, edge_weights)

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
            #print(f"obj={obj.cpu().detach().numpy():0.2f}")

        epoch_time = time.time() - start_time
        print(f"epoch {ep} t={epoch_time} total_obj={total_obj:0.2f}")

        # occasionally run validation and print loss
        if args.valid_epochs != 0 and ep % args.valid_epochs == 0:
            validation(args, model, val_loader)

        if args.save_epochs != 0 and ep % args.save_epochs == 0:
            save_model(model, f"{model_folder}/model_ep{ep}.pt")

    # save trained model
    # TODO save best model, not just a bunch of epochs.
    save_model(model, f"{model_folder}/model_ep{epochs}.pt")

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
