"""Defines the main task for the sos

The sos is defined by the following traits:
    1. 所有任务活动的能力需求都已经得到了满足
    2. 或者所选择的系统成本已经超出了成本限制，则终止该序列

Since the sos doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TaskNum=5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SoSDataset(Dataset):

    def __init__(self, size=50, Num=5 ,num_samples=1e4, seed=None):
        super(SoSDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)  
        T1=torch.randint(0,15,(num_samples,1,4))
        T2=torch.randint(0,15,(num_samples,1,4))
        T3=torch.randint(0,15,(num_samples,1,4))
        T4=torch.randint(0,15,(num_samples,1,4))        
        T5=torch.randint(0,15,(num_samples,1,4)) 

        dist1=torch.FloatTensor(num_samples,1,2).uniform_(-1,1)
        dist2=torch.FloatTensor(num_samples,1,2).uniform_(-1,1)
        dist3=torch.FloatTensor(num_samples,1,2).uniform_(-1,1)
        dist4=torch.FloatTensor(num_samples,1,2).uniform_(-1,1)
        dist5=torch.FloatTensor(num_samples,1,2).uniform_(-1,1)
        distsys=torch.FloatTensor(num_samples,2,size).uniform_(-1,1)

        dataset1 = torch.randint(0,10,(num_samples, Num-1, size)).float()
        # print(dataset1.shape)

        temp1=torch.sum(dataset1,dim=1).unsqueeze(1)
        for i in range(num_samples):
            for j in range(size):
                temp1[i][0][j]=temp1[i][0][j]*torch.rand(1)*1.8+temp1[i][0][j]
                # print(temp1[i][0][j])
                temp1[i][0][j]=temp1[i][0][j]/10
        self.dataset=torch.cat((torch.cat((dataset1,temp1),dim=1),distsys),dim=1)
        self.dataset2 = torch.zeros(num_samples, Num+Num-1+4, size*TaskNum).float()
        # print(self.dataset.shape,self.dataset2.shape)
        for i in range(num_samples):
            for j in range(size):
                # print(self.dataset2[i,:,j*TaskNum+1])
                # print(self.dataset[i,:,j])
                # print(torch.cat((T1,self.dataset[i,:,j]),dim=0))
                
                self.dataset2[i,:,j*TaskNum+0]=torch.cat((torch.cat((T1[i].squeeze(0),self.dataset[i,:,j]),dim=0),dist1[i].squeeze(0)),dim=0)
                self.dataset2[i,:,j*TaskNum+1]=torch.cat((torch.cat((T2[i].squeeze(0),self.dataset[i,:,j]),dim=0),dist2[i].squeeze(0)),dim=0)
                self.dataset2[i,:,j*TaskNum+2]=torch.cat((torch.cat((T3[i].squeeze(0),self.dataset[i,:,j]),dim=0),dist3[i].squeeze(0)),dim=0)
                self.dataset2[i,:,j*TaskNum+3]=torch.cat((torch.cat((T4[i].squeeze(0),self.dataset[i,:,j]),dim=0),dist4[i].squeeze(0)),dim=0)
                self.dataset2[i,:,j*TaskNum+4]=torch.cat((torch.cat((T5[i].squeeze(0),self.dataset[i,:,j]),dim=0),dist5[i].squeeze(0)),dim=0)
        self.dynamic = torch.zeros(num_samples, 1, size*TaskNum)
        self.num_nodes = size*TaskNum
        self.size = num_samples
        # print("T1为：",T5[-1])
        # print("static为：",self.dataset2[-1,:,-1])


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset2[idx], self.dynamic[idx], [])



def update_mask(mask, dynamic, chosen_idx,n1):
    """Marks the visited city, so it can't be selected a second time."""
    chosen_idx=chosen_idx.unsqueeze(1)
    if n1>=0:
        chose_idx=torch.ones(mask.shape[0],dynamic.shape[2]//TaskNum, device=device).long() 
        # print(chose_idx)
        for i in range(chosen_idx.shape[0]):
            for j in range(dynamic.shape[2]//TaskNum):
                chose_idx[i][j]=torch.tensor(j*TaskNum+n1)
        # print(chose_idx)

        mask.scatter_(1, chose_idx, 0)
    # print(chosen_idx.shape)
    chose_idx1=torch.ones(mask.shape[0],TaskNum, device=device).long()
    for i in range(chosen_idx.shape[0]):
        Num=chosen_idx[i][0].cpu().numpy()//TaskNum
        # print(Num)
        Num=Num[0]
        idx=torch.Tensor(range(Num*TaskNum,(Num+1)*TaskNum))
        chose_idx1[i]=idx
    # print(chose_idx1)
    mask.scatter_(1, chose_idx1, 0)
    return mask

# mask=torch.ones(1,10, device=device)
# dynamic=torch.zeros(1,1,10, device=device)
# chosen_idx=torch.tensor([[4]], device=device)
# n1=1
# print(update_mask(mask, dynamic, chosen_idx,n1))
# tour_idx1=[]
# tour_idx=[]
# a=torch.tensor([[1]])
# b=torch.tensor([[1]])
# c=torch.tensor([[1]])
# tour_idx1.append(a)
# tour_idx1.append(b)
# tour_idx1.append(c)
# print(tour_idx1)
# tour_idx1 = torch.cat(tour_idx1, dim=1)
# print(tour_idx1)
# tour_idx=torch.cat((torch.tensor(tour_idx).int(),tour_idx1), dim=0)
# print(tour_idx)
# d=torch.tensor([[2,2,2,2]])
# tour_idx = torch.cat((tour_idx,d), dim=0)
# print(tour_idx)
# print(26/5,25//5,26%5)
def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """
    commnum=torch.zeros(static.shape[0],TaskNum, device=device).long() 
    # tour_indices=tour_indices//TaskNum
    # print(tour_indices)
    tour=torch.zeros_like(tour_indices).float()
    # # Convert the indices back into a tour
    # idx = tour_indices.unsqueeze(1).expand_as(static)
    # tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    for i in range(static.shape[0]):
        for j in range (tour_indices.shape[1]):
            if tour_indices[i][j]>=0:
                tour[i][j]=static[i,8,tour_indices[i][j]]
                commnum[i,tour_indices[i][j]%TaskNum]=commnum[i,tour_indices[i][j]%TaskNum]+1

    for i in range(static.shape[0]):
        for j in range(TaskNum):
            commnum[i][j]=commnum[i][j]*(commnum[i][j]-1)/2
    reward_comm=commnum.sum(1).detach()

    # # Make a full tour by returning to the start
    # y = torch.cat((tour, tour[:, :1]), dim=1)

    # # Euclidean distance between each consecutive point
    # tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    # tour=tour_indices[:,static.shape[1]-1:static.shape[1]]
    # print(tour)

    return tour.sum(1).detach()/20 +reward_comm/200#(batch_size)

# static=torch.rand(2, 3, 4)
# print(static)
# tour_indices=torch.tensor([[2,3,-1],[1,2,-1]])
# print(reward(static, tour_indices))



def render(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)


# T1=torch.randint(0,15,(1, 4))
# T2=torch.randint(0,15,(1, 4))
# T3=torch.randint(0,15,(1, 4))
# T4=torch.randint(0,15,(1, 4))        
# T5=torch.randint(0,15,(1, 4)) 
# dataset1 = torch.randint(0,10,(1, 4, 1)).float()
# # print(dataset1.shape)

# temp1=torch.sum(dataset1,dim=1).unsqueeze(1)
# for i in range(1):
#     for j in range(1):
#         temp1[i][0][j]=temp1[i][0][j]*torch.rand(1)*0.4+temp1[i][0][j]
#         # print(temp1[i][0][j])
#         temp1[i][0][j]=temp1[i][0][j]/10
# dataset=torch.cat((dataset1,temp1),dim=1)
# dataset2 = torch.zeros(1, 9, 5).float()
# print(dataset)
# # print(self.dataset.shape,self.dataset2.shape)
# for i in range(1):
#     for j in range(1):
#         # print(self.dataset2[i,:,j*TaskNum+1])
#         # print(self.dataset[i,:,j])
#         # print(torch.cat((T1,self.dataset[i,:,j]),dim=0))
#         print(T1)
#         print(dataset[i,:,j].unsqueeze(0))
#         dataset2[i,:,j*TaskNum+0]=torch.cat((T1,dataset[i,:,j].unsqueeze(0)),dim=1)
#         print(dataset2[i,:,j*TaskNum+0])
#         dataset2[i,:,j*TaskNum+1]=torch.cat((T2,dataset[i,:,j].unsqueeze(0)),dim=1)
#         dataset2[i,:,j*TaskNum+2]=torch.cat((T3,dataset[i,:,j].unsqueeze(0)),dim=1)
#         dataset2[i,:,j*TaskNum+3]=torch.cat((T4,dataset[i,:,j].unsqueeze(0)),dim=1)
#         dataset2[i,:,j*TaskNum+4]=torch.cat((T5,dataset[i,:,j].unsqueeze(0)),dim=1)
# t=dataset2[i][:4][:5].cpu().numpy().T
# print(dataset2.shape)
# print(t)