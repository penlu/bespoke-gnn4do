import time
def train(model, train_loader, optimizer, criterion, args):
    '''Main training loop:

    Trains a model with an optimizer for a number of epochs
    
    '''
    epochs = args.epochs
    model_folder = args.log_dir

    for ep in range(epochs):
        print('epoch: ', ep)
        
        #building a new project network built on sampling a mixture of products 
        #we will train lift and project together to simplify the process
        #mixture of product abbreviated to mop
        # TODO data gen should definitely not be in here
        mode = 'mc_lift_project'
        if mode == 'mc_lift_project': 
            A, edge_index, E = gen_weighted_graph(**graph_params)
            obj,x = get_mc_obj(graph_params, conv_lift, conv_project, A, edge_index)
        elif graph_params.weight == True and graph_params.interpolate != None:
            A, edge_index, E = gen_weighted_graph(**graph_params)
            if lift_net is not None:
                obj, x = get_warm_start_obj(graph_params, conv, A, edge_index,warm_start=lift_net)
            else:
                obj, x = get_rounding_obj(graph_params, conv, A, edge_index)
        elif graph_params.weight == True and graph_params.interpolate == None:
            if lift_net is not None:
                raise NotImplementedError('warm start requires weight and interpolate parameters')
            A, edge_index, E = gen_weighted_graph(**graph_params)
            obj, x = get_weighted_obj(graph_params, conv, A, edge_index)
        else: 
            #generate a random graph
            A, edge_index, E = gen_graph(**graph_params)
            obj, x = get_obj(graph_params, conv, A, edge_index)
        
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        if ep % 100 == 0:
            with torch.no_grad():
                cut = (E - obj)/2.
                _, _, _, intcut = hyperplane_rounding(graph_params, x, A, E)
                print(f'epoch: {ep}, epoch cut: {cut}, epoch intcut: {intcut}/{E}')

                bm_obj, bm_x = try_bm(graph_params, A, edge_index, E)
                bm_cut = (E - bm_obj)/2.
                _, _, _, bm_intcut = hyperplane_rounding(graph_params, bm_x, A, E)
                print(f'epoch: {ep}, B-M comparison: cut {bm_cut}, intcut {bm_intcut}/{E}')

                #valid_obj, valid_x = get_obj(graph_params, conv, valid_A, valid_edge_index)
                valid_obj, valid_x = get_weighted_obj(graph_params, conv, valid_A, valid_edge_index)
                valid_cut = (valid_E - valid_obj)/2.
                _, _, _, valid_intcut = hyperplane_rounding(graph_params, valid_x, valid_A, valid_E)
                print(f'epoch: {ep}, valid cut: {valid_cut}, valid intcut: {valid_intcut}/{valid_E}')

                print()

        if ep % 20000 == 0:
            torch.save(conv.state_dict(), f"{model_folder}/ep{epochs}.pt")
    #torch.save(conv.state_dict(), f"{model_folder}/ep{epochs}.pt")
    #json.dump(graph_params, open(os.path.join(model_folder, 'params.txt'), 'w'))
    return model_folder

def test():
    pass

def predict():
    pass

# these three functions
# plus possibly wrangling model output
