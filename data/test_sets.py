from torch.utils.data import Dataset
from data.gset import load_gset

# provide a dataset as a list; apply transform
class ListDataset(Dataset):
    def __init__(self, data,
                  transform=None, pre_transform=None, pre_filter=None):
        # apply pre_filter
        if pre_filter is not None:
            data = [x for x in data if pre_filter(x)]

        # apply pre_transform
        if pre_transform is not None:
            data = [pre_transform(x) for x in data]

        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index])
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def construct_kamis_dataset(args, pre_transform=None, transform=None):
    pickled_data = pickle.load(open('/home/penlu/code/bespoke-gnn4do/graphs_and_results.pickle', 'rb'))
    #dataset_names = ['er', 'ba', 'hk', 'ws']
    #for ds in dataset_names:
    #    datapoints = [y[0] for y in pickled_data[ds].values()]
    #    test_loader = DataLoader(datapoints, batch_size=args.batch_size, shuffle=False)
    # TODO add args for dataset names
    return ListDataset([y[0] for y in pickled_data['er'].values()])

def construct_gset_dataset(args, pre_transform=None, transform=None):
    # TODO add args for dataset names
    return ListDataset(load_gset('datasets/GSET'))
