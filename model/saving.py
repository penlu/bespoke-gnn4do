import torch
# functions for saving and loading models
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    # might need to specify map location?
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


