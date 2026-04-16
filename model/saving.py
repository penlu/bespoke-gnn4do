import torch
# functions for saving and loading models
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    # weights_only=True restricts unpickling to a safe allowlist, preventing
    # arbitrary code execution from a malicious checkpoint. state_dicts only
    # contain tensors, so this does not affect legitimate checkpoints.
    state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    return model


