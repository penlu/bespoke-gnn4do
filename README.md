# What this is

You have found the repository for [Are Graph Neural Networks Optimal Approximation Algorithms?](https://arxiv.org/abs/2310.00526)!

# Dependencies

Minimally you will need to install:
- [PyTorch](https://pytorch.org/) (tested with versions 2.0.0 and 2.0.1)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) (tested with versions 2.3.0 and 2.3.1)
- [NetworkX](https://networkx.org/documentation/stable/install.html)

To run baselines, we require:
- [CVXPy](https://www.cvxpy.org/install/index.html)
- [Gurobi](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python)
- [Mosek](https://docs.mosek.com/latest/pythonapi/install-interface.html)

# How to run it

## Training

The output of `python train.py --help` is fairly up-to-date. In general, run with a command of the form:
```
python train.py --prefix=[directory] ...
```
Output files from the training run, such as parameters used and model checkpoints, will appear in `training_runs/[directory]/paramhash:[hash]`.

Here are three example command lines, with commentary.

```
python train.py --prefix=example-1 --problem_type=max_cut \
                --dataset=ErdosRenyi --gen_n=100 --gen_p=0.15 \
                --model_type=LiftMP --num_layers=16 --rank=16 \
                --stepwise=True --steps=1000 --valid_freq=500 --batch_size=32
```
- The prefix is `example`, so output will appear in `training_runs/example/paramhash:[hash]`.
- The dataset is `ErdosRenyi`, so these graphs will be generated. We will generate 1000 (by default). Each will contain `--gen_n=100` nodes, and the edge probability is `--gen_p=0.15`.
- The model type is `LiftMP`, which is the paper architecture. We have specified 16 layers and rank 16, which means we will maintain a 16-dimensional embedding on each node.
- We have specified `--stepwise=True`, so the training duration is controlled by `--steps=1000` instead of `--epochs`. We will validate every `--valid_freq=500` steps.
- We want a batch size of `--batch_size=32`. This is the first thing to try adjusting when training runs out of memory.

```
python train.py --stepwise=True --steps=10000 \
                --valid_freq=1000 --dropout=0 \
                --prefix=example-2 \
                --model_type=LiftMP --dataset=ErdosRenyi --parallel=20 --infinite=True --gen_n 50 100 \
                --num_layers=16 --rank=32 \
                --penalty=1 --problem_type=vertex_cover \
                --batch_size=16 --positional_encoding=random_walk --pe_dimension=8
```
- We turn on infinite data with `--infinite=True`. This means that instead of pre-generating and reusing a fixed number of graphs, we generate each batch on the fly.
- We use `--parallel=20` to specify the use of 20 worker processes for data generation.
- Each of the `--gen_[nmkp]` flags can accept two parameters. Here we see `--gen_n 50 100`, which means each graph's node count will be sampled uniformly at random from `[50, 100]`.
- Some problems, such as `vertex_cover`, include constraints, which require a `--penalty` for weighting.
- We're using a positional encoding with `--positional_encoding=random_walk`. When using a positional encoding, also specify the dimensionality of the encoding with `--pe_dimension`.

```
python train.py --stepwise=True --steps=1000 --valid_freq=1000 \
                --prefix=example-3 \
                --dataset=random-sat --infinite=True --parallel=20 --gen_n=100 --gen_k 400 430
                --model_type=LiftMP --num_layers=16 --rank=32 \
                --penalty=0.003 --problem_type=sat \
                --batch_size=32 --dropout=0
```
In this example, we're training a LiftMP for random 3-SAT with 100 variables and 400 to 430 clauses (selected uniformly at random per instance).

### Specifying a dataset

You can specify the desired training dataset using a combination of flags. The function `add_dataset_args` in `utils/parsing.py` names the relevant flags. Dataset construction and handling is mostly in the `data/` folder; see especially `data/loader.py`.

Using a TU dataset is most straightforward; just provide `--dataset=[name]`. We currently allow the following TU datasets:
- COLLAB
- ENZYMES
- IMDB-BINARY
- MUTAG
- PROTEINS
- REDDIT-BINARY
- REDDIT-MULTI-5K
- REDDIT-MULTI-12K

We support the generated datasets:
- ErdosRenyi (supply `--gen_n` and `--gen_p`)
- BarabasiAlbert (supply `--gen_n` and `--gen_m`)
- PowerlawCluster (supply `--gen_n`, `--gen_m`, and `--gen_p`)
- WattsStrogatz (supply `--gen_n`, `--gen_k`, and `--gen_p`)
- ForcedRB (supply `--gen_n`, `--gen_k`, and `--gen_p`)
- random-sat (supply `--gen_n`, `--gen_k`, and `--gen_p`)

### Full `--help` output

```
$ python train.py --help
usage: train.py [-h] [--problem_type {max_cut,vertex_cover,max_clique,sat}] [--seed SEED] [--prefix PREFIX] [--model_type {LiftMP,FullMP,GIN,GAT,GCNN,GatedGCNN,NegationGAT,ProjectMP,Nikos}] [--num_layers NUM_LAYERS]
                [--repeat_lift_layers REPEAT_LIFT_LAYERS [REPEAT_LIFT_LAYERS ...]] [--num_layers_project NUM_LAYERS_PROJECT] [--rank RANK] [--dropout DROPOUT] [--hidden_channels HIDDEN_CHANNELS] [--norm NORM] [--heads HEADS]
                [--finetune_from FINETUNE_FROM] [--lift_file LIFT_FILE] [--lr LR] [--batch_size BATCH_SIZE] [--valid_freq VALID_FREQ] [--save_freq SAVE_FREQ] [--penalty PENALTY] [--stepwise STEPWISE] [--steps STEPS] [--epochs EPOCHS]
                [--train_fraction TRAIN_FRACTION]
                [--dataset {ErdosRenyi,BarabasiAlbert,PowerlawCluster,WattsStrogatz,ForcedRB,ENZYMES,PROTEINS,IMDB-BINARY,MUTAG,COLLAB,REDDIT-MULTI-5K,REDDIT-MULTI-12K,REDDIT-BINARY,random-sat,kamis,gset}] [--data_seed DATA_SEED]
                [--parallel PARALLEL] [--num_graphs NUM_GRAPHS] [--infinite INFINITE] [--gen_n GEN_N [GEN_N ...]] [--gen_m GEN_M [GEN_M ...]] [--gen_k GEN_K [GEN_K ...]] [--gen_p GEN_P [GEN_P ...]]
                [--positional_encoding {laplacian_eigenvector,random_walk}] [--pe_dimension PE_DIMENSION] [--split_seed SPLIT_SEED]

options:
  -h, --help            show this help message and exit
  --problem_type {max_cut,vertex_cover,max_clique,sat}
                        What problem are we doing?
  --seed SEED           Torch random seed to use to initialize networks
  --prefix PREFIX       Folder name for run outputs if desired (will default to run timestamp)
  --model_type {LiftMP,FullMP,GIN,GAT,GCNN,GatedGCNN,NegationGAT,ProjectMP,Nikos}
                        Which type of model to use
  --num_layers NUM_LAYERS
                        How many layers?
  --repeat_lift_layers REPEAT_LIFT_LAYERS [REPEAT_LIFT_LAYERS ...]
                        A list of the number of times each layer should be repeated
  --num_layers_project NUM_LAYERS_PROJECT
                        How many projection layers? (when using FullMP)
  --rank RANK           How many dimensions for the vectors at each node, i.e. what rank is the solution matrix?
  --dropout DROPOUT     Model dropout
  --hidden_channels HIDDEN_CHANNELS
                        Dimensions of the hidden channels
  --norm NORM           Normalization to use
  --heads HEADS         number of heads for GAT
  --finetune_from FINETUNE_FROM
                        model file to load weights from for finetuning
  --lift_file LIFT_FILE
                        model file from which to load lift network
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch size for training
  --valid_freq VALID_FREQ
                        Run validation every N steps/epochs (0 to never run validation)
  --save_freq SAVE_FREQ
                        Save model every N steps/epochs (0 to only save at end of training)
  --penalty PENALTY     Penalty for constraint violations
  --stepwise STEPWISE   Train by number of gradient steps or number of epochs?
  --steps STEPS         Training step count
  --epochs EPOCHS       Training epoch count
  --train_fraction TRAIN_FRACTION
                        Fraction of data to retain for training. Remainder goes to validation/testing.
  --dataset {ErdosRenyi,BarabasiAlbert,PowerlawCluster,WattsStrogatz,ForcedRB,ENZYMES,PROTEINS,IMDB-BINARY,MUTAG,COLLAB,REDDIT-MULTI-5K,REDDIT-MULTI-12K,REDDIT-BINARY,random-sat,kamis,gset}
                        Dataset type to use
  --data_seed DATA_SEED
                        Seed to use for generated datasets (RANDOM and ForcedRB)
  --parallel PARALLEL   How many parallel workers to use for generating data?
  --num_graphs NUM_GRAPHS
                        When using generated datasets, how many graphs to generate? (Ignored when using --infinite)
  --infinite INFINITE   When using generated datasets, do infinite generation?
  --gen_n GEN_N [GEN_N ...]
                        Range for the n parameter of generated dataset (usually number of vertices)
  --gen_m GEN_M [GEN_M ...]
                        m parameter of generated dataset (meaning varies)
  --gen_k GEN_K [GEN_K ...]
                        Range for the k parameter of generated dataset (meaning varies)
  --gen_p GEN_P [GEN_P ...]
                        p parameter of generated dataset (meaning varies)
  --positional_encoding {laplacian_eigenvector,random_walk}
                        Use a positional encoding?
  --pe_dimension PE_DIMENSION
                        Dimensionality of the positional encoding
  --split_seed SPLIT_SEED
                        Seed to use for train/val/test split
```

## Testing

To test, use `python test.py`.
- To identify the model to test, supply the `--model_folder` containing the model's `params.txt`, and the `--model_file` indicating which checkpoint to use.
- Specify a dataset for testing using the same flags discussed above.
- The outputs will be placed in the same directory as the model, with a file prefix as specified by `--test_prefix`.

Here is an example invocation:
```
python test.py --model_folder="training_runs/example-1/paramhash:[hash]" --model_file=best_model.pt \
               --problem_type=sat --dataset=random-sat --infinite=True --gen_n 100 --gen_k 400 \
               --test_prefix=sat_test
```

# Contact

Contact us via the issue tracker on this repository, or via our emails as listed in the arXiv preprint.

# License

Copyright 2024 the authors.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
