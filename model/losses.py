# loss functions
# and their associated gradient functions

# TODO go through the following
import matplotlib.pyplot as plt

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_torch_csr_tensor

import cvxpy as cp

# TODO
def get_loss_fn(args):
    if args.problem_type == 'max_cut':
        return max_cut_lift_loss
    elif args.problem_type == 'vertex_cover':
        raise NotImplementedError('vertex_cover loss not yet implemented')
    elif args.problem_type == 'max_clique':
        raise NotImplementedError('max_clique loss not yet implemented')

def max_cut_lift_loss(X, edge_index):
    # compute lifted loss
    A = to_torch_csr_tensor(edge_index)
    XX = torch.matmul(X, torch.transpose(X, 0, 1))
    obj = torch.trace(torch.matmul(A, XX)) / 2.

    return obj

def max_cut_project_loss(X, edge_index):
    #apply random rotation 
    #with supervision of lifted loss rotation should not be necessary
    #without lifted loss, rotation enforces rotational symmetry of lift net output
    #this is critical as project net must have an enumerative search component
    #manifested by repeated random rotations.  

    #pass through project net 
    x_project = conv_project(x_lift,edge_index,edge_weights) 

    #compute projected loss  
    xx_project = torch.matmul(x_project, torch.transpose(x_project, 0, 1))
    project = torch.trace(torch.matmul(A, xx_project)) / (2.*graph_params.r)
    print('lifted loss: ', lift)
    print('projected loss: ', project)

    #interpolate represents the relative weight of the lifted vs projected loss
    #default setting 0.5
    obj = (1-interpolate)*project + interpolate*lift
    return obj, x_project

#loss functions of max cut
def objective_lift_mc(graph_params, conv_mc, A, edge_index, warm_start=None, interpolate=0.5, mode=None):
    '''calculates the standard dot product loss 
    interpolated with the one dimensional loss
    Input:
    graph_params
    conv : the nn to be run
    A: Adjacency matrix
    edge_index: 2xE adjacency list for MessagePassing
    interpolate: when equal to 1.0 the loss is the vector loss 
    Returns: 
    obj: loss
    x : the output after application of the nn
    '''
    
    conv_lift = conv_mc['lift']
    #generate random vectors
    x_in = torch.randn((graph_params.N, graph_params.r), dtype=torch.float, device=graph_params.device)
    x_in = F.normalize(x_in, dim=1)
    #feed forward through lift net

    num_edge = edge_index.shape[1]
    edge_weights = torch.ones(num_edge, device=graph_params.device)
    for col in range(num_edge):
        (i,j) = edge_index[:,col]
        edge_weights[col] = A[i,j]
    
    #pass through lift net 
    x_lift = conv_lift(x_in, edge_index, edge_weights)
    #x_lift = F.normalize(x_lift, dim=1)
    
    #compute lifted loss
    xx_lift = torch.matmul(x_lift, torch.transpose(x_lift, 0, 1))
    lift = torch.trace(torch.matmul(A, xx_lift)) / 2.
    
    #apply random rotation 
    #with supervision of lifted loss rotation should not be necessary
    #without lifted loss, rotation enforces rotational symmetry of lift net output
    #this is critical as project net must have an enumerative search component
    #manifested by repeated random rotations.  
    
    print('lifted loss: ', lift)
    
    #certification code, default mode is None
    with torch.no_grad(): 
        if mode=='certify':
            #print('x_lift normalize: ', F.normalize(x_lift))
            #gradient 
            grad_v = torch.matmul(A,x_lift)
            #print('grad V normalize: ', F.normalize(grad_v))
            #key heuristic for extracting dual:
            dual_var = 0.5*torch.norm(grad_v,dim=1)
            #print('dual_var: ', dual_var)
            grad_X = 0.5*A + torch.diag(dual_var)
            #ensure symmetric
            grad_X = 0.5*(grad_X + torch.transpose(grad_X,0,1))
            #print('grad_X: ', grad_X)
            eigvals,_ = torch.linalg.eigh(grad_X)
            #print('eigvals: ', eigvals)
            #print('min eigval: ', torch.min(eigvals))

            duality_gap = torch.abs(lift + torch.sum(dual_var))
            #print('duality gap: ',  duality_gap)
            certificate = lift + graph_params.N*torch.min(eigvals) - duality_gap
            print('certified lower bound: ', certificate)
        else: 
            certificate = None
        
    
    #interpolate represents the relative weight of the lifted vs projected loss
    #default setting 0.5
    obj = lift
    return obj, x_lift, certificate

def objective_lift_project_mc(graph_params, conv_mc, A, edge_index, warm_start=None,interpolate=0.5):
    '''calculates the standard dot product loss 
    interpolated with the one dimensional loss
    Input:
    graph_params
    conv : the nn to be run
    A: Adjacency matrix
    edge_index: 2xE adjacency list for MessagePassing
    interpolate: when equal to 1.0 the loss is the vector loss 
    Returns: 
    obj: loss
    x : the output after application of the nn
    '''
    
    conv_lift = conv_mc['lift']
    conv_project = conv_mc['project']
    #generate random vectors
    x_in = torch.randn((graph_params.N, graph_params.r), dtype=torch.float, device=graph_params.device)
    x_in = F.normalize(x_in, dim=1)
    #feed forward through lift net

    num_edge = edge_index.shape[1]
    edge_weights = torch.ones(num_edge, device=graph_params.device)
    for col in range(num_edge):
        (i,j) = edge_index[:,col]
        edge_weights[col] = A[i,j]
    
    #pass through lift net 
    x_lift = conv_lift(x_in, edge_index, edge_weights)
    #x_lift = F.normalize(x_lift, dim=1)
    
    #compute lifted loss
    xx_lift = torch.matmul(x_lift, torch.transpose(x_lift, 0, 1))
    lift = torch.trace(torch.matmul(A, xx_lift)) / 2.
    
    #apply random rotation 
    #with supervision of lifted loss rotation should not be necessary
    #without lifted loss, rotation enforces rotational symmetry of lift net output
    #this is critical as project net must have an enumerative search component
    #manifested by repeated random rotations.  
    
    
    #pass through project net 
    x_project = conv_project(x_lift,edge_index,edge_weights) 
    
    
    #compute projected loss  
    xx_project = torch.matmul(x_project, torch.transpose(x_project, 0, 1))
    project = torch.trace(torch.matmul(A, xx_project)) / (2.*graph_params.r)
    print('lifted loss: ', lift)
    print('projected loss: ', project)
    
    #interpolate represents the relative weight of the lifted vs projected loss
    #default setting 0.5
    obj = (1-interpolate)*project + interpolate*lift
    return obj, x_project

