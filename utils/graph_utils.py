import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx, to_networkx
import time

# TODO perhaps want to just be working with networkx graphs
def gen_graph(N=100, p=0.15, device=torch.device("cpu"), **kwargs):
    ''' Generates a random graph
        TODO(penlu,morrisy): doublecheck definitions
        N: number of vertices
        p: edge probability

        Returns: tuple of
        A : Adjacency matrix
        edge_index : 2xE adjacency list for MessagePassing
        E : number of edges 
    '''
    # generate random graph (erdos-renyi)
    A = torch.bernoulli(torch.full((N, N), p, device=device).triu(diagonal=1))

    # make it symmetric
    A = A + torch.transpose(A, 0, 1)

    # number of edges
    E = torch.sum(A)/2

    # edge_index is 2xE adjacency list for MessagePassing
    edge_index = dense_to_sparse(A)[0]

    return A, edge_index, E

def random_orthogonal_matrix(d):
    """
    Generate a random orthogonal matrix of shape (d, d).
    """
    a = torch.randn(d, d)
    q, _ = torch.qr(a)
    return q

def random_rotation(graph_params, conv_mc, A, edge_index,iterate=10):
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
    
    #apply random rotation 
    #with supervision of lifted loss rotation should not be necessary
    #without lifted loss, rotation enforces rotational symmetry of lift net output
    #this is critical as project net must have an enumerative search component
    #manifested by repeated random rotations.  
    E = torch.sum(A)
    #the qr decomposition here could be slow, consider revising 
    N = graph_params.N
    r = graph_params.r
    cuts = torch.zeros(r*iterate)
    index = 0
    for i in range(iterate):
        orthogonal_matrix = random_orthogonal_matrix(r)
        #make sure dimensions match 
        x_lift = torch.matmul(x_lift,orthogonal_matrix)
        #pass through project net 
        x_project = conv_project(x_lift,edge_index,edge_weights) 
        x_sign = torch.sign(x_project)
        for i in range(r):
            spins = torch.outer(x_sign[:,i], x_sign[:,i])
            cuts[index] = (E - torch.trace(torch.matmul(A,spins))/2.)/2. 
            index = index + 1
    
    return torch.max(cuts)

complement_time = 0.
# XXX caution: this drops attributes!
def complement_graph(G):
    global complement_time
    start_time = time.time()
    nx_graph = to_networkx(G)
    nx_complement = nx.operators.complement(nx_graph)
    new_graph = from_networkx(nx_complement)
    complement_time += time.time() - start_time
    print(f"complement_graph: {complement_time}")
    return new_graph
