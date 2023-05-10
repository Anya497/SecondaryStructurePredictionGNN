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
        return list(map(lambda x: "dataset_1091/in/" + x, os.listdir("./data3/raw/dataset_1091/in")))

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
           
            x = torch.zeros([196, 4], dtype=torch.float32)
            edge_index = [[], []]
            edge_weight = []
            pos_edge_index = [[], []]
            neg_edge_index = [[], []]
            
            in_im = imread(raw_path, as_gray=True)
            out_im = imread(raw_path.replace("in", "out"), as_gray=True)
            size = len(in_im)
            
            adj_mat = torch.zeros([196, 196])


            for i in range(size):
                for j in range(size):
                    
                    #adj_mat
                    if out_im[i][j] == 255:
                        adj_mat[i][j] = 1
                        adj_mat[j][i] = 1
                        
                        pos_edge_index[0].append(i)
                        pos_edge_index[1].append(j)  
                        pos_edge_index[0].append(j)
                        pos_edge_index[1].append(i) 
            
                        
                    #edge_index
                    if in_im[i][j] == 255:
                        edge_index[0].append(i)
                        edge_index[1].append(j)
                        edge_weight.append([1, 0])
                        
                        edge_index[0].append(j)
                        edge_index[1].append(i)
#                         edge_weight.append([1, 0])
                        
                    if (j == (i + 1)):
                        edge_index[0].append(i)
                        edge_index[1].append(j) 
                        edge_weight.append([0, 1])
                        
                        edge_index[0].append(j)
                        edge_index[1].append(i)
#                         edge_weight.append([0, 1])
                    
                    #neg_edge_index
                    if (in_im[i][j] == 255) and (out_im[i][j] != 255):
                        neg_edge_index[0].append(i)
                        neg_edge_index[1].append(j)
                        neg_edge_index[0].append(j)
                        neg_edge_index[1].append(i)    
                        
            #x
            for i in range(size):
                nucl_ind = in_im[i][i]
                if nucl_ind == 32:
                    x[i] = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
                elif nucl_ind == 64:
                    x[i] = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
                elif nucl_ind == 96:
                    x[i] = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
                elif nucl_ind == 128:
                    x[i] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
                    
#             x = torch.tensor(x, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
            neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
            
            
            data = Data(x=x, edge_index=edge_index.contiguous(), edge_weight=edge_weight, adj_mat=adj_mat, 
                        pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index, seq_size=size)

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

    
dataset = dataset = RNADataset(root="./data3/")

