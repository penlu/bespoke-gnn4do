import numpy as np
import os
import time

import torch
import torch.nn.functional as F

from model.saving import save_model
from problem.baselines import random_hyperplane_projector

from torch_geometric.transforms import AddRandomWalkPE

def featurize_batch(args, batch):
    batch = batch.to(args.device)

    N = batch.num_nodes
    num_edges = batch.edge_index.shape[1]

    # build x
    if args.positional_encoding is None or args.pe_dimension == 0:
        # generate random vector input
        x_in = torch.randn((N, args.rank), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
    elif args.positional_encoding == 'laplacian_eigenvector':
        x_in = torch.randn((N, args.rank - args.pe_dimension), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
        pe = batch.laplacian_eigenvector_pe.to(args.device)[:, :args.pe_dimension]
        sign = -1 + 2 * torch.randint(0, 2, (args.pe_dimension, ), device=args.device)
        pe *= sign
        x_in = torch.cat((x_in, pe), 1)
    elif args.positional_encoding == 'random_walk':
        # XXX add the random walk PE here
        if not hasattr(batch, 'random_walk_pe'):
            batch = AddRandomWalkPE(walk_length=args.pe_dimension)(batch.to(args.device))
        x_in = torch.randn((N, args.rank - args.pe_dimension), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
        pe = batch.random_walk_pe.to(args.device)[:, :args.pe_dimension]
        x_in = torch.cat((x_in, pe), 1)
    else:
        raise ValueError(f"Invalid transform passed into featurize_batch: {args.transform}")

    # TODO handling multi-penalty situations -- shouldn't be in featurize
    batch.penalty = args.penalty

    return x_in, batch

# measure and return the validation loss
def validate(args, model, val_loader, problem):
    total_loss = 0.
    total_score = 0.
    total_constraint = 0.
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 1:
                datalist = [batch]
            else:
                datalist = batch.to_data_list()

            x_in, batch = featurize_batch(args, batch)
            x_out = model(x_in, batch)
            loss = problem.loss(x_out, batch)

            total_loss += float(loss)

            x_proj = random_hyperplane_projector(args, x_out, batch, problem.score)

            # ENSURE we are getting a +/- 1 vector out by replacing 0 with 1
            x_proj = torch.where(x_proj == 0, 1, x_proj)

            num_zeros = (x_proj == 0).count_nonzero()
            assert num_zeros == 0

            # count the score
            score = problem.score(args, x_proj, batch)
            total_score += float(score)
            total_constraint += float(problem.constraint(x_proj, batch))

            total_count += len(batch)

    return total_loss / total_count, total_score / total_count, total_constraint / total_count

def train(args, model, train_loader, optimizer, problem, val_loader=None, test_loader=None):
    '''Main training loop:

    Trains a model with an optimizer for a number of epochs
    '''
    epochs = args.epochs
    model_folder = args.log_dir

    train_losses = []
    valid_losses = []
    valid_scores = []
    valid_constraints = []
    test_losses = []
    test_scores = []
    test_constraints = []

    model.to(args.device)

    ep = 0
    steps = 0

    while args.stepwise or ep < epochs:
        start_time = time.time()

        # reset epoch average loss counters
        epoch_total_loss = 0.
        epoch_count = 0

        for batch in train_loader:
            # run the model
            x_in, batch = featurize_batch(args, batch)
            x_out = model(x_in, batch)

            # get loss
            loss = problem.loss(x_out, batch)

            # run gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate and store average loss for batch
            avg_loss = float(loss) / batch.num_graphs
            train_losses.append(avg_loss)

            # increment epoch loss counters
            epoch_total_loss += float(loss)
            epoch_count += batch.num_graphs

            steps += 1
            if args.stepwise:
                # occasionally print training loss for infinite datasets
                if args.infinite and steps % 100 == 0:
                    epoch_time = time.time() - start_time
                    epoch_avg_loss = epoch_total_loss / epoch_count
                    print(f"steps={steps} t={epoch_time:0.2f} epoch_avg_loss={epoch_avg_loss:0.2f}")

                    start_time = time.time()
                    epoch_total_loss = 0.
                    epoch_count = 0

                # occasionally run validation
                if args.valid_freq != 0 and steps % args.valid_freq == 0:
                    valid_start_time = time.time()
                    valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem)
                    # save model if it's the current best
                    if len(valid_scores)==0 or valid_score > max(valid_scores):
                        save_model(model, f"{model_folder}/best_model.pt")
                    valid_losses.append(valid_loss)
                    valid_scores.append(valid_score)
                    valid_constraints.append(valid_constraint)
                    valid_time = time.time() - valid_start_time
                    
                    # test
                    if test_loader is not None:
                        test_loss, test_score, test_constraint = validate(args, model, test_loader, problem)
                        test_losses.append(test_loss)
                        test_scores.append(test_score)
                        test_constraints.append(test_constraint)
                    else:
                        test_loss = np.inf
                        test_score = -np.inf

                    print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f}\n\
                                valid_loss={valid_loss} valid_score={valid_score} valid_constraint={valid_constraint}\n\
                                test_loss={test_loss} test_score={test_score} test_constraint={test_constraint}")

                # check if training is done
                if steps >= args.steps:
                    break

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
                valid_start_time = time.time()
                valid_loss, valid_score = validate(args, model, val_loader, problem)
                valid_losses.append(valid_loss)
                valid_scores.append(valid_score)
                valid_time = time.time() - valid_start_time
                print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f} valid_loss={valid_loss} valid_score={valid_score}")

            # occasionally save model
            if args.save_freq != 0 and ep % args.save_freq == 0:
                save_model(model, f"{model_folder}/model_ep{ep}.pt")

        ep += 1

    # end of training: save trained model
    # TODO save best model, not just a bunch of epochs
    if not args.stepwise:
        save_model(model, f"{model_folder}/model_ep{epochs}.pt")
    else:
        save_model(model, f"{model_folder}/model_step{steps}.pt")
    np.save(os.path.join(args.log_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(args.log_dir, "valid_losses.npy"), valid_losses)
    np.save(os.path.join(args.log_dir, "valid_scores.npy"), valid_scores)
    np.save(os.path.join(args.log_dir, "valid_constraints.npy"), valid_constraints)
    np.save(os.path.join(args.log_dir, "test_losses.npy"), test_losses)
    np.save(os.path.join(args.log_dir, "test_scores.npy"), test_scores)
    np.save(os.path.join(args.log_dir, "test_constraints.npy"), test_constraints)


def predict(model, loader, args):
    batches = []
    # TODO decide return signature and transform.
    for batch in loader:
        assert False, "not adapted to new featurize_batch: proceed with caution!"
        x_in, edge_index, edge_weight = featurize_batch(args, batch)
        x_out = model(x_in, edge_index, edge_weight)
        batches.append((x_out, edge_index))
    return batches
