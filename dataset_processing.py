import os
import os.path as osp
import glob
from skimage.io import imread

import torch
from torch_geometric.data import Dataset, Data


class RNADataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return list(map(lambda x: "dataset_1091/in/" + x, os.listdir("./data/raw/dataset_1091/in")))

    @property
    def processed_file_names(self):
        l = []
        for i in range(1091):
            l.append("data_" + str(i) + ".pt")
        return l

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        # ...
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            in_im_arr = imread(raw_path, as_gray=True)
            out_im_arr = imread(raw_path.replace("in", "out"), as_gray=True)
            size = len(in_im_arr)

            nodes = []
            edges = []

            out_edges = []
            edge_label_index = []

            for i in range(size):
                l = []
                for j in range(size):
                    if out_im_arr[i][j] == 255:
                        l.append(1)
                    else:
                        l.append(0)
                edge_label_index.append(l)

            for i in range(size):
                nucl_ind = in_im_arr[i][i]
                if nucl_ind == 32:
                    nodes.append([1, 0, 0, 0])
                elif nucl_ind == 64:
                    nodes.append([0, 1, 0, 0])
                elif nucl_ind == 96:
                    nodes.append([0, 0, 1, 0])
                elif nucl_ind == 128:
                    nodes.append([0, 0, 0, 1])

                if i != size - 1:
                    edges.append([i, i + 1])
                    edges.append([i + 1, i])
                    out_edges.append([i, i + 1])
                    out_edges.append([i + 1, i])

                    edge_label_index[i][i + 1] = 1
                    edge_label_index[i + 1][i] = 1

                for j in range(i, size):
                    if in_im_arr[i][j] == 255:
                        edges.append([i, j])
                        edges.append([j, i])
                    if out_im_arr[i][j] == 255:
                        out_edges.append([i, j])
                        out_edges.append([j, i])

                        edge_label_index[i][j] = 1
                        edge_label_index[j][i] = 1

            x = torch.tensor(nodes, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
            # out_edge_index = torch.tensor(out_edges, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_label_index=edge_label_index)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            print(data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
