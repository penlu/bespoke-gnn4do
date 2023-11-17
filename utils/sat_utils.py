from itertools import product
import time
from torch import tensor
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from math import ceil
from matplotlib import pyplot as plt
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
import networkx as nx
from torch.nn import Parameter
from torch_geometric.utils import degree
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean, scatter, scatter_softmax
from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
from torch_geometric.utils import is_undirected
import scipy
import scipy.io
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import pickle
from pysat.formula import CNF
import pysat as ps
import numpy as np
import os

def old_dense_to_sparse(tensor):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        tensor (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert tensor.dim() == 2
    index = tensor.nonzero(as_tuple=False).t().contiguous()
    value = tensor[index[0], index[1]]
    return index, value            
                        

def generate_random_kCNF(k,num_variables, num_clauses):
    formula = CNF()
    for clause_i in range(num_clauses):
        clause = []
        for literal in range(k):
            #randint is not inclusive on the upper bound, hence +1
            literal = np.random.randint(-num_variables,num_variables+1)
            while (literal in clause) or (-literal in clause) or (literal==0):
                    literal = np.random.randint(-num_variables,num_variables+1)
            clause += [literal]
        formula.append(clause)
    return formula

#detect dependencies between clauses (2 clauses have a dependency if they share a variable)
#dependencies are lopsided, a variable and its negative will connect two clauses with a negative sign
def detect_dependency_2(clause_1, clause_2):
    dependency_score = 0
    for  var_1 in clause_1:
        for var_2 in clause_2:
            if np.abs(var_1) == np.abs(var_2):
                if np.sign(var_1)==np.sign(var_2):
                    return 1
                else:
                    return -1
            
    return 0


def detect_dependency(clause_1, clause_2):
    dependency_score = 0
    for  var_1 in clause_1:
        for var_2 in clause_2:
            if np.abs(var_1) == np.abs(var_2):
                if np.sign(var_1)==np.sign(var_2):
                    dependency_score = 1
                else:
                    return -1
            
    return dependency_score

def get_VarClause_matrix(formula,k):
    clause_var_adj = np.zeros((len(formula.clauses)+formula.nv,len(formula.clauses)+formula.nv))
    clause_var_dir = np.zeros((formula.nv,formula.nv+len(formula.clauses)))
    for i in range(len(formula.clauses)):
        for j in range(k):
            literal = formula.clauses[i][j] 
            clause_var_adj[i,np.abs(literal)-1] =  np.sign(literal)
            clause_var_dir[np.abs(literal)-1,i+formula.nv] = np.sign(literal)
    return clause_var_adj, clause_var_dir
    
def get_VarClause_matrix_2(formula,k):
    clause_var_adj = np.zeros((len(formula.clauses)+formula.nv,len(formula.clauses)+formula.nv))
    clause_var_dir = np.zeros((formula.nv,formula.nv+len(formula.clauses)))
    for i in range(len(formula.clauses)):
        for j in range(k):
            literal = formula.clauses[i][j] 
            clause_var_adj[i,np.abs(literal)-1] =  np.sign(literal)
            clause_var_dir[np.abs(literal)-1,i+formula.nv] = np.sign(literal)
    return clause_var_adj, clause_var_dir
    
    
  
def get_dependency_matrix(formula):
    num_clauses = len(formula.clauses)
    dependency_graph_adj = np.zeros((len(formula.clauses)+formula.nv,len(formula.clauses)+formula.nv))
    for row in range(num_clauses):
        for column in range(num_clauses):
            dependency_graph_adj[row+formula.nv,column+formula.nv] = detect_dependency(formula.clauses[row], formula.clauses[column])
    return dependency_graph_adj- np.eye(dependency_graph_adj.shape[0])

def generate_RandomCNFDataset(num_graphs, k, num_variables_low, num_clauses_low, num_variables_high=None, num_clauses_high=None):
    dataset=[]
    for i in range(num_graphs):
        if num_variables_high is None:
            num_variables = num_variables_low
        else:
            num_variables = np.random.randint(num_variables_low, num_variables_high)
            
        if num_clauses_high is None:
            num_clauses = num_clauses_low 
        else:
            num_clauses = np.random.randint(num_clauses_low, num_clauses_high)

        num_variables = np.random.randint(num_variables_low, num_variables_high)        
        formula = generate_random_kCNF(k, num_variables,num_clauses)

        
        with Lingeling(bootstrap_with=formula.clauses, with_proof=True) as l:
            is_sat = (l.solve())
            
        num_variables = len(set([np.abs(item) for sublist in (formula.clauses) for item in sublist]))
        num_clauses  = len(formula.clauses)

        adj_matrix = get_dependency_matrix(formula)
        var_clause_mat, var_clause_dirmat = get_VarClause_matrix(formula,k)
    
        
        #delete isolated node rows,columns from clause-clause graph and var-clause graph
        adj_matrix = np.delete(adj_matrix,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=0)
        adj_matrix = np.delete(adj_matrix,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=1)


        var_clause_dirmat_temp = np.delete(var_clause_dirmat,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=0)
        var_clause_dirmat_temp = np.delete(var_clause_dirmat_temp,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=1)

        var_clause_dirmat = var_clause_dirmat_temp


        #print(var_clause_dirmat.shape)
        vc_edge_ind, vc_edge_attr = old_dense_to_sparse(torch.tensor(var_clause_dirmat))
        #dirvc_edge_ind, dirvc_edge_attr = dense_to_sparse(torch.tensor(var_clause_dirmat))
        #print("it's undirected: ", is_undirected(vc_edge_ind))
        x_var = torch.zeros(num_variables)
        x_clause = torch.ones(num_clauses)
        x_comb = torch.cat([x_var,x_clause])
        cc_edge_ind, cc_edge_attr = old_dense_to_sparse(torch.tensor(adj_matrix))
        
#         if contains_isolated_nodes(vc_edge_ind):
#             print(np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0])
#             print("contains isolated nodes")
#             breakpoint()
        graph_object = Data(x= x_comb, edge_index = vc_edge_ind, edge_attr = vc_edge_attr, cc_edge_index = cc_edge_ind, cc_edge_att =cc_edge_attr,  formula = [formula],  sat=is_sat)
        dataset += [graph_object]
        print("current graph: ", i)
    return dataset
    
    
    
#PLOTTING UTILS    
def create_plottable_object(data, value_dict, select_graph=4):
    selected_index = (data.batch == select_graph)

    clause_selected_index = data.batch[data.x==1]
    clause_selected_index = [clause_selected_index==select_graph]

    selected_x  = data.x[selected_index]
    data_r, data_c = data.edge_index
    data_er, data_ec = data.cc_edge_index
    
    offset = ((data.batch < select_graph)*1.).sum()

    num_vars = (selected_x==0).sum().item()
    num_clauses = (selected_x==1).sum().item()
    
    selected_eindex = (data.batch[data_r] == select_graph)
    selected_edge_index = data.edge_index[:,selected_eindex] - offset
    selected_edge_attr = data.edge_attr[selected_eindex]
    sr, sc = selected_edge_index 

    
    selected_cindex = (data.batch[data_er] == select_graph)
    selected_cedge_index = data.cc_edge_index[:,selected_cindex] - offset
    selected_cedge_attr = data.cc_edge_att[selected_cindex]
    ser, sec = selected_cedge_index 
    selected_cedge_index, selected_cedge_attr = remove_self_loops(selected_cedge_index, selected_cedge_attr)
    
    
    clause_probs_full = value_dict["clause_probs_full"][0]
    var_probs_full = value_dict["variable_probs_full"][0]

    selected_clause_probs = clause_probs_full[selected_index]
    selected_clause_probs = selected_clause_probs[selected_x==1]


    selected_var_probs = var_probs_full[selected_index]
    selected_var_probs = selected_var_probs[selected_x==0]

    selected_x_i = value_dict["x_i"][0]
    selected_x_i = selected_x_i[selected_index]
    selected_x_i = selected_x_i[selected_x==1]

    upper_bounds = value_dict["upper bounds"][0]
    selected_upper_bounds = upper_bounds[clause_selected_index]

    violated_clauses = (torch.where(selected_clause_probs > selected_upper_bounds)[0]+num_vars).tolist()
    feasible_clauses = (torch.where(selected_clause_probs <= selected_upper_bounds)[0]+num_vars).tolist()
    
    #print(selected_x[violated_clauses],selected_x[feasible_clauses])
    
    event_graph_object = Data(x=selected_x, edge_index = selected_cedge_index.long(), edge_attr = selected_cedge_attr)
    event_graph = to_networkx(event_graph_object)
    
    data_object = Data(x = selected_x, edge_index = selected_edge_index.long(), edge_attr = selected_edge_attr, clause_probs = selected_clause_probs.detach().cpu(), var_probs= selected_var_probs.detach().cpu(), x_i = selected_x_i.detach().cpu(), feasible_clauses = feasible_clauses ,violated_clauses = violated_clauses, cedge_attr = selected_cedge_attr, cedge_index = selected_cedge_index, event_graph = event_graph)
    
    return data_object


def plot_vis_data(data_object):
    g = to_networkx(data_object)

    e_pos = graphviz_layout(data_object.event_graph)

    num_vars = (data_object.x == 0).sum()
    num_clauses = (data_object.x== 1).sum()
    pos = graphviz_layout(g)


    #pos = nx.spring_layout(g, scale=10., dim = 3)
    var_keys = list(pos.keys())[:num_vars]
    clause_keys = list(pos.keys())[num_vars:]


    e_clause_keys = list(e_pos.keys())[num_vars:]


    var_pos = {key: pos[key] for key in var_keys }
    clause_pos = {key: pos[key] for key in clause_keys }
    feasible_clause_pos = {key: clause_pos[key] for key in data_object.feasible_clauses}
    violated_clause_pos = {key: clause_pos[key] for key in data_object.violated_clauses}

    e_clause_pos = {key: e_pos[key] for key in e_clause_keys }
    e_feasible_clause_pos = {key: e_clause_pos[key] for key in data_object.feasible_clauses}
    e_violated_clause_pos = {key: e_clause_pos[key] for key in data_object.violated_clauses}

    var_nodes = g.subgraph(var_keys)
    clause_nodes = g.subgraph(clause_keys)
    feasible_clause_nodes = g.subgraph(data_object.feasible_clauses)
    violated_clause_nodes = g.subgraph(data_object.violated_clauses)

    e_clause_nodes = data_object.event_graph.subgraph(e_clause_keys)
    e_feasible_clause_nodes = data_object.event_graph.subgraph(data_object.feasible_clauses)
    e_violated_clause_nodes = data_object.event_graph.subgraph(data_object.violated_clauses)

    edge_list = list(g.edges())
    positive_edges = [edge_list[k] for k in torch.where(data_object.edge_attr>0)[0].tolist()]
    negative_edges = [edge_list[k] for k in torch.where(data_object.edge_attr<0)[0].tolist()] 


    e_edge_list = list(data_object.event_graph.edges())
    e_positive_edges = [e_edge_list[k] for k in torch.where(data_object.cedge_attr>0)[0].tolist()]
    e_negative_edges = [e_edge_list[k] for k in torch.where(data_object.cedge_attr<0)[0].tolist()]



    adjusted_feasible_clauses = [k-num_vars.cpu().item() for k in data_object.feasible_clauses]
    adjusted_violated_clauses = [k-num_vars.cpu().item() for k in data_object.violated_clauses]

    
    
    plt.figure(1,figsize=(36,20))
    #nx.draw_networkx_nodes(feasible_clauses, feasible_clause_pos, node_color = 'g', node_shape='*',alpha=1., node_size = 2000);
    
    #draw violated clauses
    nx.draw_networkx_nodes(violated_clause_nodes, violated_clause_pos, node_color = 'r',  node_shape='x',alpha=1., node_size = [np.exp(10*prob)*2000  for prob in data_object.clause_probs[adjusted_violated_clauses]]);

    #draw variable probs
    nx.draw_networkx_nodes(var_nodes, var_pos, node_shape='o',alpha=0.75, node_size = [prob*500+200 for prob in data_object.var_probs]);
    
    #draw x_i
    nx.draw_networkx_nodes(clause_nodes, clause_pos, node_shape='s', node_color= 'black', node_size = [np.exp(10*x_i)*1000 for x_i in data_object.x_i],alpha=0.75);
    
    #draw labels x_i
    nx.draw_networkx_labels(clause_nodes, clause_pos, labels = {n:np.round(lab.item(),2) for n,lab in zip(clause_pos.keys(),data_object.x_i)}, font_color = 'white', font_size=10, font_weight= 1000);

    
    #draw event probs
    nx.draw_networkx_nodes(clause_nodes, clause_pos, node_shape='s', node_color = [plt.cm.bwr(prob.item()/data_object.clause_probs.max().item()) for prob in data_object.clause_probs], node_size = [np.exp(10*prob)*1000 for prob in data_object.clause_probs],alpha=0.75);



    #draw positive edges 
    nx.draw_networkx_edges(g,pos, width = 3.0,  edgelist  = positive_edges, edge_color = 'black', style = 'solid', alpha = 0.5);
    
    #draw negative edges
    nx.draw_networkx_edges(g,pos, width = 3.0,  edgelist  = negative_edges, edge_color='black', style = 'dashed', alpha = 0.5);


    #print(adjusted_feasible_clauses)
    
    #this is for the event graph, looks pretty bad for now

#     plt.figure(2,figsize=(36,20))
#     nx.draw_networkx_nodes(e_feasible_clause_nodes, e_feasible_clause_pos, node_color= 'green', node_shape='s', alpha = 0.75, node_size = [prob*5000+200 for prob in data_object.clause_probs[adjusted_feasible_clauses]])
#     nx.draw_networkx_nodes(e_violated_clause_nodes, e_violated_clause_pos, node_shape='s', node_color= 'red', alpha = 0.75, node_size = [prob*5000+200 for prob in data_object.clause_probs[adjusted_violated_clauses]])
#     nx.draw_networkx_nodes(e_clause_nodes, e_clause_pos, node_color= 'black', alpha = 0.75, node_shape='s', node_size = [x_i*5000+200 for x_i in data_object.x_i])


#     nx.draw_networkx_edges(e_clause_nodes, e_clause_pos, width = 3.0, edgelist = e_positive_edges, edge_color='g', alpha = 0.5);
#     nx.draw_networkx_edges(e_clause_nodes, e_clause_pos, width = 3.0, edgelist = e_negative_edges, edge_color='r', alpha = 0.5);



def convert_dataset_to_dimacs(dataset):
#Write to DIMACS format:
    for (counter,current_data) in enumerate(dataset):
        with open('/mnt/scratch/lts2/karalias/repoz/datasets/dimacs/3sat_dimacs/' + "graph_no"+str(counter)+ '_.dimacs', 'a+') as the_file:
            formula = current_data.formula[0]
            num_clauses = (current_data.x==1).sum().item()
            num_variables =  (current_data.x==0).sum().item()
            the_file.write('p cnf '+ str(num_variables)+ ' ' + str(num_clauses)+ '\n')
            for clause in range((num_clauses)):
                for literal in formula.clauses[clause]:
                    the_file.write(str(literal) + ' ')
                the_file.write(str(0) +'\n')
 
def convert_to_dimacs(dataset):
    for (k,data) in enumerate(dataset):
        data.formula[0].to_file("data"+str(k)+"_"+str(data.sat*1)+".cnf")

def read_dimacs_directory(directory):
    directory_contents = os.listdir(directory)
    dataset_formulas = []
    for k in range(len(directory_contents)):
        if "cnf" in directory_contents[k].split()[0]:
            formula = CNF(from_file= directory + '/' +directory_contents[k])
            dataset_formulas += [formula]  


def generate_dimacs_CNFDataset(dimacs_dataset, k):
    dataset=[]
    
    for (counter,formula) in enumerate(dimacs_dataset):

        with Glucose4(bootstrap_with=formula.clauses, with_proof=True) as l:
            is_sat = (l.solve())
            
        num_variables = len(set([np.abs(item) for sublist in (formula.clauses) for item in sublist]))
        num_clauses  = len(formula.clauses)

        adj_matrix = get_dependency_matrix(formula)
        var_clause_mat, var_clause_dirmat = get_VarClause_matrix(formula,k)
        
        #delete isolated node rows,columns from clause-clause graph and var-clause graph
        adj_matrix = np.delete(adj_matrix,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=0)
        adj_matrix = np.delete(adj_matrix,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=1)


        var_clause_dirmat_temp = np.delete(var_clause_dirmat,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=0)
        var_clause_dirmat_temp = np.delete(var_clause_dirmat_temp,np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0],axis=1)

        var_clause_dirmat = var_clause_dirmat_temp
        
                #print(var_clause_dirmat.shape)
        vc_edge_ind, vc_edge_attr = old_dense_to_sparse(torch.tensor(var_clause_dirmat))
        #dirvc_edge_ind, dirvc_edge_attr = dense_to_sparse(torch.tensor(var_clause_dirmat))
        #print("it's undirected: ", is_undirected(vc_edge_ind))
        x_var = torch.zeros(num_variables)
        x_clause = torch.ones(num_clauses)
        x_comb = torch.cat([x_var,x_clause])
        cc_edge_ind, cc_edge_attr = old_dense_to_sparse(torch.tensor(adj_matrix))
        
#         if contains_isolated_nodes(vc_edge_ind):
#             print(np.where((np.abs(var_clause_dirmat).sum(1)[:formula.nv]==0))[0])
#             print("contains isolated nodes")
#             breakpoint()
        graph_object = Data(x= x_comb, edge_index = vc_edge_ind, edge_attr = vc_edge_attr, cc_edge_index = cc_edge_ind, cc_edge_att =cc_edge_attr,  formula = [formula],  sat=is_sat)
        dataset += [graph_object]
        print("current graph: ", counter)
        
    return dataset

def detect_dependency_new(clause_small, clause_big):
    dependence = 0
    for var in clause_small:
        if -var in clause_big:
            return -1
        elif var in clause_big:
            dependence = 1
        
    return dependence

def get_dependency_matrix_new_set(formula):
    formula.sets = [set(clause) for clause in formula.clauses]
    num_sets = len(formula.sets)
    
    dependency_graph_adj = np.zeros((len(formula.sets)+formula.nv,len(formula.sets)+formula.nv))
    for row in range(num_sets):
        for column in range(row, num_sets):
            dependence = detect_dependency_new(formula.sets[row], formula.sets[column])
            dependency_graph_adj[row+formula.nv,column+formula.nv] = dependence
            dependency_graph_adj[column+formula.nv, row+formula.nv] = dependence

    return dependency_graph_adj- np.eye(dependency_graph_adj.shape[0])

def get_dependency_matrix_new(formula):
    num_clauses = len(formula.clauses)
    dependency_graph_adj = np.zeros((len(formula.clauses)+formula.nv,len(formula.clauses)+formula.nv))
    for row in range(num_clauses):
        for column in range(row, num_clauses):
            dependence = detect_dependency_new(formula.clauses[row], formula.clauses[column])
            dependency_graph_adj[row+formula.nv,column+formula.nv] = dependence
            dependency_graph_adj[column+formula.nv, row+formula.nv] = dependence

    return dependency_graph_adj- np.eye(dependency_graph_adj.shape[0])


def detect_dependency_new(clause_small, clause_big):
    dependence = 0
    for var in clause_small:
        if -var in clause_big:
            return -1
        elif var in clause_big:
            dependence = 1
        
    return dependence

def distributed_indset2(data, x_new=None):
    #initialization
    updated_w_loops = add_self_loops(data.cc_edge_index)
    clr, clc = updated_w_loops[0]
    graph_select_clauses = (data.x==1)
    num_vars = (data.x==0).sum()
    cr, cc = data.cc_edge_index    
    indset=torch.tensor([],device='cuda')
    if x_new==None:
        x_1 = torch.rand_like(data.x)
    else:
        x_1 = x_new.detach()+ torch.rand_like(x_new.detach())*1e-6
    independence_test = 0.
    curr_it = 0
    batch_arange = torch.arange(data.x.shape[0])
    batch_arange_clauses = batch_arange[graph_select_clauses]
    #print("hello")

    #while set is independent
    while (True):
        x_1_clauses = x_1[graph_select_clauses]
        #generate mask for current indset and its neighbors
        x_mask = torch.zeros_like(x_1)
        x_mask_clauses = x_mask[graph_select_clauses]
        x_mask_clauses[indset.long()] = 1
        x_mask[graph_select_clauses] = x_mask_clauses
        x_mask_scatter = scatter(x_mask[clr],clc,reduce='max')
        x_mask = (x_mask_scatter>0)
        spike_value = torch.abs(x_1.sum()) #float("Inf")# torch.abs(x_1.max())
        x_1_clauses[x_mask[graph_select_clauses]] = spike_value
        
        #do message passing to find new indset nodes
        x_1[graph_select_clauses] = x_1_clauses
        phase_x = scatter(x_1[clr],clc,reduce='min')
        prev_indset = indset.detach()
        indset= torch.cat([torch.where(~(x_mask[graph_select_clauses]) * (phase_x[graph_select_clauses]==x_1[graph_select_clauses]))[0], indset])   
        curr_it+=1
        if (indset.numel()-prev_indset.numel())==0:
            break
    indset = batch_arange_clauses[indset.long()]
    #print(indset)
#     if ((subgraph(indset.long(), data.cc_edge_index)[0]).numel())>0:
#         print("INVALID: SET NOT INDEPENDENT")
# #         breakpoint()

    return indset