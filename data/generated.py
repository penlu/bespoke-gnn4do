import itertools
import numpy as np

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import InMemoryDataset

class GeneratedDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=1000, seed=0, parallel=0,
                  transform=None, pre_transform=None, pre_filter=None,
                  **kwargs):
        self.num_graphs = num_graphs
        self.seed = seed
        self.parallel = parallel

        self.kwargs = kwargs

        super(GeneratedDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        raise NotImplementedError('Please override this method.')

    def generate(self, seed, **kwargs):
        raise NotImplementedError('Please override this method.')

    def process(self):
        # initialize RNG
        worker = get_worker_info()
        if worker == None:
            seed = self.seed
        else:
            # if we are a worker, we generate a seed
            # as written this should put out 256 bits: collision is unlikely
            seed = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker.id,)).generate_state(8)

        # generate graphs and save
        # TODO doing this in parallel when requested
        data_list = list(itertools.islice(self.generate(seed, **self.kwargs), self.num_graphs))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GeneratedIterableDataset(IterableDataset):
    def __init__(self, seed=0,
                  transform=None, pre_transform=None, pre_filter=None,
                  **kwargs):
        self.seed = seed

        self.kwargs = kwargs

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(GeneratedIterableDataset, self).__init__()

    def generate(self, seed, **kwargs):
        raise NotImplementedError('Please override this method.')

    def __iter__(self):
        # compute our seed
        worker = get_worker_info()
        if worker == None:
            seed = self.seed
        else:
            # if we are a worker, we generate a seed
            # as written this should put out 256 bits: collision is unlikely
            seed = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker.id,)).generate_state(8)

        # generate graphs forever
        for G in self.generate(seed, **self.kwargs):
            # apply filters and transforms
            if self.pre_filter is not None and not self.pre_filter(G):
                continue
            if self.pre_transform is not None:
                G = self.pre_transform(G)
            if self.transform is not None:
                G = self.transform(G)

            yield G
