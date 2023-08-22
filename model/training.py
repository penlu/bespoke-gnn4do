import numpy as np
import os
import time

import torch
import torch.nn.functional as F

from model.saving import save_model

def featurize_batch(args, batch):
    N = batch.num_nodes
    edge_index = batch.edge_index.to(args.device)

    if args.transform is None:
        # generate random vector input
        x_in = torch.randn((N, args.rank), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
    elif args.transform == 'laplacian_eigenvector_PE':
        x_in = torch.randn((N, args.rank - args.eigenvector_k), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
        pe = batch.laplacian_eigenvector_pe.to(args.device)
        x_in = torch.cat((x_in, pe), 1)
    else:
        raise ValueError(f"Invalid transform passed into featurize_batch: {args.transform}")

    # run model
    # TODO later, a more robust attribute system
    num_edges = edge_index.shape[1]
    edge_weight = torch.ones(num_edges, device=args.device)
    return x_in, edge_index, edge_weight

# measure and return the validation loss
def validate(args, model, val_loader, criterion):
    total_loss = 0.
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            x_in, edge_index, edge_weight = featurize_batch(args, batch)
            x_out = model(x=x_in, edge_index=edge_index, edge_weight=edge_weight)
            loss = criterion(x_out, edge_index)
            total_loss += loss.cpu().detach().numpy()
            total_count += batch.num_graphs
    return total_loss / total_count

def train(args, model, train_loader, optimizer, criterion, val_loader=None):
    '''Main training loop:

    Trains a model with an optimizer for a number of epochs
    '''
    epochs = args.epochs
    model_folder = args.log_dir

    train_losses = []
    valid_losses = []

    model.to(args.device)

    steps = 0

    for ep in range(epochs):
        start_time = time.time()

        # reset epoch average loss counters
        epoch_total_loss = 0.
        epoch_count = 0

        for batch in train_loader:
            # run the model
            x_in, edge_index, edge_weight = featurize_batch(args, batch)
            x_out = model(x=x_in, edge_index=edge_index, edge_weight=edge_weight)

            # get objective
            loss = criterion(x_out, edge_index)

            # run gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate and store average loss for batch
            avg_loss = loss.cpu().detach().numpy() / batch.num_graphs
            train_losses.append(avg_loss)
            #print(f"avg_loss={avg_loss:0.2f}")

            # increment epoch loss counters
            epoch_total_loss += loss.cpu().detach().numpy()
            epoch_count += batch.num_graphs

            steps += 1
            if args.stepwise:
                # check if training is done
                if steps >= args.steps:
                    break

                # occasionally run validation
                if args.valid_freq != 0 and steps % args.valid_freq == 0:
                    valid_loss = validate(args, model, val_loader, criterion)
                    valid_losses.append(valid_loss)
                    print(f"  VALIDATION steps={steps} valid_loss={valid_loss}")

                # occasionally save model
                if args.save_freq != 0 and steps % args.save_freq == 0:
                    save_model(model, f"{model_folder}/model_step{steps}.pt")

        if args.stepwise and steps >= args.steps:
            break

        # print average loss for epoch
        epoch_time = time.time() - start_time
        epoch_avg_loss = epoch_total_loss / epoch_count
        print(f"epoch={ep} t={epoch_time:0.2f} steps={steps} epoch_avg_loss={epoch_avg_loss:0.2f}")

        if not args.stepwise:
            # occasionally run validation
            if args.valid_freq != 0 and ep % args.valid_freq == 0:
                valid_loss = validate(args, model, val_loader, criterion)
                valid_losses.append(valid_loss)
                print(f"  VALIDATION epoch={ep} steps={steps} valid_loss={valid_loss}")

            # occasionally save model
            if args.save_freq != 0 and ep % args.save_freq == 0:
                save_model(model, f"{model_folder}/model_ep{ep}.pt")

    # end of training: save trained model
    # TODO save best model, not just a bunch of epochs
    if not args.stepwise:
        save_model(model, f"{model_folder}/model_ep{epochs}.pt")
    else:
        save_model(model, f"{model_folder}/model_step{steps}.pt")
    np.save(os.path.join(args.log_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(args.log_dir, "valid_losses.npy"), valid_losses)

def predict(model, loader, args):
    batches = []
    # TODO decide return signature and transform.
    for batch in loader:
        x_in, edge_index, edge_weight = featurize_batch(args, batch)
        x_out = model(x_in, edge_index, edge_weight)
        batches.append((x_out, edge_index))
    return batches
