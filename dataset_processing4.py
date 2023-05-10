import os
import os.path as osp
import glob
from skimage.io import imread

import torch
from torch_geometric.data import Dataset, Data


# codes = {'A': 32, 'C': 64, 'G': 96, 'U': 128, 'T': 128}

class RNADataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return list(map(lambda x: "dataset_1091/in/" + x, os.listdir("./data5/raw/dataset_1091/in")))

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
           
            x = []
            edge_index = [[], []]
            
            in_im = imread(raw_path, as_gray=True)
            out_im = imread(raw_path.replace("in", "out"), as_gray=True)
            y = []
            size = len(in_im)
            
            adj_mat = torch.zeros([size, size])

            #x            
            for i in range(size):
                n1 = in_im[i][i]               
                for j in range(i, size):
                    n2 = in_im[j][j]
                    if abs(i - j) >= 3:
                        # AU
                        if (n1 == 32 and n2 == 128) or (n1 == 128 and n2 == 32):
                            if in_im[i][j] == 255:
                                x.append([1, 0, 0, 1, 0, [i, j]]) 
                            else:
                                x.append([1, 0, 0, 0, 1, [i, j]]) 

                            if out_im[i][j] == 255:
                                y.append(1)
                            else:
                                y.append(0)
#                         # UA
#                         elif n1 == 128 and n2 == 32:
#                             if in_im[i][j] == 255:
#                                 x.append([0, 1, 0, 0, 0, 0, 1, 0, [i, j]]) 
#                             else:
#                                 x.append([0, 1, 0, 0, 0, 0, 0, 1, [i, j]]) 

#                             if out_im[i][j] == 255:
#                                 y.append(1)
#                             else:
#                                 y.append(0)
                        # CG
                        elif (n1 == 64 and n2 == 96) or (n1 == 96 and n2 == 64):
                            if in_im[i][j] == 255:
                                x.append([0, 1, 0, 1, 0, [i, j]]) 
                            else:
                                x.append([0, 1, 0, 0, 1, [i, j]]) 

                            if out_im[i][j] == 255:
                                y.append(1)
                            else:
                                y.append(0)
#                         # GC
#                         elif n1 == 96 and n2 == 64:
#                             if in_im[i][j] == 255:
#                                 x.append([0, 0, 0, 1, 0, 0, 1, 0, [i, j]]) 
#                             else:
#                                 x.append([0, 0, 0, 1, 0, 0, 0, 1, [i, j]]) 

#                             if out_im[i][j] == 255:
#                                 y.append(1)
#                             else:
#                                 y.append(0)
                        # UG
                        elif (n1 == 128 and n2 == 96) or (n1 == 96 and n2 == 128):
                            if in_im[i][j] == 255:
                                x.append([0, 0, 1, 1, 0, [i, j]]) 
                            else:
                                x.append([0, 0, 1, 0, 1, [i, j]]) 

                            if out_im[i][j] == 255:
                                y.append(1)
                            else:
                                y.append(0)
                        # GU
#                         elif n1 == 96 and n2 == 128:
#                             if in_im[i][j] == 255:
#                                 x.append([0, 0, 0, 0, 0, 1, 1, 0, [i, j]]) 
#                             else:
#                                 x.append([0, 0, 0, 0, 0, 1, 0, 1, [i, j]]) 

#                             if out_im[i][j] == 255:
#                                 y.append(1)
#                             else:
#                                 y.append(0)
                    if j - i == 1:
                        x.append([0, 0, 0, 0, 0, [i, j]])
                        y.append(-1)
                 
            #edge_index
            for i in range(len(x)):
                for j in range(len(x)):
                    if ((x[i][-1][0] in x[j][-1]) or (x[i][-1][1] in x[j][-1])) and (i != j):
                        edge_index[0].append(i)
                        edge_index[1].append(j)
            
            
            for i in range(len(x)):
                x[i].pop()
                
            x = torch.tensor(x, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.float32)
            
            data = Data(x=x, edge_index=edge_index.contiguous(), y=y, adj_mat=adj_mat, 
                        # pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index
                        )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


dataset = dataset = RNADataset(root="./data5/")

