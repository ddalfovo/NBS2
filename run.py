#!/usr/bin/env python
# coding: utf-8

# import numpy as np
# import pandas as pd
import SRW_v1 as SRW
# import pickle
# import functools
# from multiprocessing import Pool, cpu_count
import time
# import torch.multiprocessing as mp
from importlib import reload  # Python 3.4+

# def get_cuda_memory(element, size = 'GB'):
#     elBytes = element.element_size() * element.nelement()
#     match size:
#         case 'MB':
#             conversion = 1024*1024
#         case 'GB':
#             conversion = 1024*1024*1024
#         case _:
#             conversion = 1
#     return elBytes/conversion

# import subprocess as sp
# import os

# def get_gpu_memory():
#     command = "nvidia-smi --query-gpu=memory.free --format=csv"
#     memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
#     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#     return memory_free_values


tumor = 'TCGA'
netUsed = 'PCNet'
# folderSomaticAlt = 'test'
folderSomaticAlt = 'test'
typeMut = 'cna'

edges, features, node_names = SRW.load_network('../data/'+folderSomaticAlt+'/'+tumor+'_edge2features_'+typeMut+'_'+netUsed+'.txt')

P_init_train, sample_names_train = SRW.load_samples('../data/'+folderSomaticAlt+'/'+tumor+'_alterations_'+typeMut+'_training.tsv', node_names)
P_init_val, sample_names_val = SRW.load_samples('../data/'+folderSomaticAlt+'/'+tumor+'_alterations_'+typeMut+'_validation.tsv', node_names)
group_labels_train = SRW.load_grouplabels('../data/'+folderSomaticAlt+'/'+tumor+'_labels_'+typeMut+'_training.tsv')
group_labels_val = SRW.load_grouplabels('../data/'+folderSomaticAlt+'/'+tumor+'_labels_'+typeMut+'_validation.tsv')

# first = ["BRCA","HNSC","STAD","LUSC","LUAD","BLCA","UCS","ESCA","OV","SARC"]
# second = ["ACC","KICH"]
# third = ["TGCT","COADREAD","SKCM","CESC","LIHC"]
# fourth = ["DLBC","CHOL","GBM","MESO","PRAD","PAAD"]
# fifth = ["KIRP","UCEC","LGG","KIRC","UVM","THYM","PCPG","LAML","THCA"]
# dictTCGA = {}
# for i in first:
#     dictTCGA[i] = 'class1'

# for i in second:
#     dictTCGA[i] = 'class2'

# for i in third:
#     dictTCGA[i] = 'class3'

# for i in fourth:
#     dictTCGA[i] = 'class4'

# for i in fifth:
#     dictTCGA[i] = 'class5'

# group_labels_train = [dictTCGA[i] for i in group_labels_train]
# group_labels_val = [dictTCGA[i] for i in group_labels_val]

# import random
# group_labels_train = random.sample(group_labels_train,len(group_labels_train))
# group_labels_val = random.sample(group_labels_val,len(group_labels_val))

feature_names = []
with open('../data/'+folderSomaticAlt+'/'+tumor+'_feature_names_'+typeMut+'.txt') as f:
    for line in f.read().rstrip().splitlines():
        feature_names.append(line)


feature_names.append('selfloop')
feature_names.append('intercept')
nnodes = len(node_names)

### Put the optimized from 3-fold cross validation step
rst_prob_fix = 0.3 # alpha
lam_fix = 1e-1 # delta
beta_loss_fix = 2e-2 # beta



reload(SRW)
SRW_obj = SRW.SRW_solver(edges, features, nnodes, P_init_train, rst_prob_fix, group_labels_train, lam_fix, 
                         w_init_sd=0.01, w=None, feature_names=feature_names, 
                         sample_names=sample_names_train, node_names=node_names, loss='WMW', 
                         norm_type='L1', learning_rate=0.1, update_w_func='Adam', 
                         P_init_val=P_init_val, group_labels_val=group_labels_val, ncpus=10, 
                         maxit=2, early_stop=2, WMW_b=beta_loss_fix)
SRW_obj.train_SRW_GD()



# t=0
# SRW_obj.init_w()
# w_local = SRW_obj.w
############################# CUDA
##################################
def logistic_edge_strength_cuda(features, w):
    return  1.0 / (1+torch.exp(-features.mv(w)))

def logistic_strength_gradient_cuda(features, edge_strength):
    logistic_slop = torch.multiply(edge_strength, (1-edge_strength))[:,np.newaxis]
    return features.multiply(logistic_slop)

def renorm_neg_cuda(M):
    M = M.to_dense()
    M_sum = abs(M).sum(dim=1,keepdim=True)
    M_norm = torch.div(M,(M_sum + 1e-8))
    return M_norm.to_sparse_csr().to(torch.float64)

def strength_Q_and_gradient_cuda(edges, nnodes, features, w):
    # Calculate edge strength and the gradient of strength
    edge_strength = logistic_edge_strength_cuda(features, w)
    strength_grad = logistic_strength_gradient_cuda(features, edge_strength)
    # M_strength (n by n) is a matrix containing edge strength
    # where M[i,j] = strength[i,j];
    M_strength = torch.sparse_coo_tensor(
        indices=torch.stack([edges[:, 0], edges[:, 1]]),
        values=edge_strength,
        size=(nnodes, nnodes))
    M_strength_rowSum = torch.sparse.sum(M_strength, dim=1)
    # Normalize the transition matrix
    Q = renorm_neg_cuda(M_strength)
    return Q, M_strength, M_strength_rowSum, strength_grad

def Q_gradient_1feature_cuda(edges, nnodes, M_strength, M_strength_rowSum, strength_grad):
    # M_strength_grad (n by n) is a matrix containing the gradient of edge strength
    # where M_strength_grad[i,j] = strength_grad[i,j];
    M_strength_grad = torch.sparse_coo_tensor(
        indices=torch.stack([edges[:, 0], edges[:, 1]]),
        values=torch.squeeze(strength_grad),
        size=(nnodes, nnodes))
    M_strength_grad_rowSum = torch.sparse.sum(M_strength_grad,dim=1)
    Q_grad = (M_strength_grad.to_dense().T.multiply(M_strength_rowSum.to_dense()) 
              -M_strength.to_dense().T.multiply(M_strength_grad_rowSum.to_dense())) / torch.square(M_strength_rowSum).to_dense()
    return Q_grad.T.to_sparse_csr()

def Q_gradient_1feature_cudaBatch(edges, nnodes, M_strength, M_strength_rowSum, strength_grad_batch):
    # M_strength_grad (n by n) is a matrix containing the gradient of edge strength
    # where M_strength_grad[i,j] = strength_grad[i,j];
    strength_grad_batch = strength_grad.reshape(1,-1)
    indices = torch.stack([edges[:, 0], edges[:, 1]])
    idx = torch.arange(strength_grad.shape[1]).repeat_interleave(indices.shape[1])
    indices = torch.cat((indices.repeat(1, strength_grad.shape[1]), idx.unsqueeze(0)))
    #
    M_strength_grad = torch.sparse_coo_tensor(
            indices=indices,
            values=torch.squeeze(strength_grad_batch),
            size=(nnodes, nnodes, strength_grad.shape[1]))
    M_strength_grad_rowSum = torch.sparse.sum(M_strength_grad,dim=1)
    #
    Q_grad = (M_strength_grad.to_dense().T.multiply(M_strength_rowSum.to_dense()) 
              -M_strength.to_dense().T.multiply(M_strength_grad_rowSum.to_dense())) / torch.square(M_strength_rowSum).to_dense()
    #
    # M_strength_grad = torch.sparse_coo_tensor(
    #     indices=torch.stack([edges[:, 0], edges[:, 1]]),
    #     values=torch.squeeze(strength_grad_batch),
    #     size=(nnodes, nnodes))
    # M_strength_grad_rowSum = torch.sparse.sum(M_strength_grad,dim=1)
    # Q_grad = (M_strength_grad.to_dense().T.multiply(M_strength_rowSum.to_dense()) 
    #           -M_strength.to_dense().T.multiply(M_strength_grad_rowSum.to_dense())) / torch.square(M_strength_rowSum).to_dense()
    return Q_grad.T.to_sparse_csr()

def allclose_cuda(a, b, rtol=1e-5, atol = 1e-8):
    c = abs(a-b) - rtol*abs(b)
    return (c.max() <= atol).item()

def iterative_P_gradient_1feature_cuda(P, Q, Q_grad, rst_prob):
    # Initlalize P_grad to be all zeros. See below for P_grad_1_iter
    P_grad = torch.zeros(P.shape, dtype=torch.float64)
    P_dot_Qgrad = torch.mm(P,Q_grad)
    P_grad_new = P_grad_1feature_1iter_cuda(rst_prob, P_grad, Q, P_dot_Qgrad)
    # Iteratively calculate P_grad until converged
    while not(allclose_cuda(P_grad, P_grad_new)):
        P_grad = P_grad_new
        P_grad_new = P_grad_1feature_1iter_cuda(rst_prob, P_grad, Q, P_dot_Qgrad)
    return P_grad_new

def iterative_PPR_cuda(Q, P_init, rst_prob):
    # Q and P_init are already normalized by row sums
    # Takes P_init and a transition matrix to find the PageRank of nodes
    P = P_init
    rst_prob_P_init = rst_prob*P_init.to_dense()
    P_new = (1-rst_prob)*torch.sparse.mm(P,Q).to_dense() + rst_prob_P_init
    # P_new =  (1-rst_prob)*cp.dot(P, Q) + rst_prob_P_init
    while not(allclose_cuda(P.to_dense(), P_new)):
        P = P_new
        P_new =  (1-rst_prob)*torch.sparse.mm(P.to_sparse_csr(), Q).to_dense() + rst_prob_P_init
    return P_new

def iterative_PPR_conv_cuda(Q, P_init, rst_prob):
    # Q and P_init are already normalized by row sums
    # Takes P_init and a transition matrix to find the PageRank of nodes
    term1 = rst_prob*P_init.to_dense()
    term2 = torch.eye(P_init.shape[1])-(1-rst_prob)*Q.to_dense()
    term2_inv = torch.linalg.inv(term2)
    P_new = torch.mm(term1, term2_inv)
    return P_new

def iterative_PPR_conv_cuda_solve(Q, P_init, rst_prob):
    # Q and P_init are already normalized by row sums
    # Takes P_init and a transition matrix to find the PageRank of nodes
    term1 = rst_prob*P_init.to_dense()
    term2 = torch.eye(P_init.shape[1])-(1-rst_prob)*Q.to_dense()
    # term2_inv = torch.linalg.inv(term2)
    # P_new = torch.mm(term1, term2_inv)
    P_new = torch.linalg.solve(term2.T, term1.T).T
    return P_new

def iterative_P_gradient_1feature_cuda(P, Q, Q_grad, rst_prob):
    # Initlalize P_grad to be all zeros. See below for P_grad_1_iter
    P_grad = torch.zeros(P.shape, dtype=torch.float64)
    P_dot_Qgrad = torch.mm(P,Q_grad)
    P_grad_new = P_grad_1feature_1iter_cuda(rst_prob, P_grad, Q, P_dot_Qgrad)
    print("enter the while")
    # Iteratively calculate P_grad until converged
    while not(allclose_cuda(P_grad, P_grad_new)):
        print(".")
        P_grad = P_grad_new
        P_grad_new = P_grad_1feature_1iter_cuda(rst_prob, P_grad, Q, P_dot_Qgrad)
    return P_grad_new

def P_grad_1feature_1iter_cuda(rst_prob, P_grad, Q, P_dot_Qgrad):
    return (1-rst_prob) * (torch.mm(P_grad,Q.to_dense()) + P_dot_Qgrad)


def iterative_P_gradient_1feature_conv_cuda(P, Q, Q_grad, rst_prob, term2):
    # P_grad = torch.zeros(P.shape, dtype=torch.float64).cuda()
    term1 = (1-rst_prob)*torch.mm(P,Q_grad)
    # P_grad_new = torch.mm(term1, term2_inv)
    P_grad_new = torch.linalg.solve(term2.T,term1.T).T
    return P_grad_new


def iterative_P_gradient_1feature_conv_cuda(P, Q, Q_grad, rst_prob, term2):
    # P_grad = torch.zeros(P.shape, dtype=torch.float64).cuda()
    term1 = (1-rst_prob)*torch.mm(P.cuda(),Q_grad.cuda())
    # P_grad_new = torch.mm(term1, term2_inv)
    P_grad_new = torch.linalg.solve(term2.T,term1.T).T
    return P_grad_new.cpu()


def calc_P_grad_1fea_cuda(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, term2,
                     strength_grad):
    Q_grad = Q_gradient_1feature_cuda(edges, nnodes, M_strength.to_dense(), M_strength_rowSum.to_dense(),
                                 strength_grad)
    # P_grad = iterative_P_gradient_1feature_cuda(P, Q, Q_grad, rst_prob)
    P_grad = iterative_P_gradient_1feature_conv_cuda(P, Q, Q_grad, rst_prob, term2)
    return P_grad.numpy()

def calc_P_grad_cuda(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, 
                     strength_grad, ncpus=-1):
    # Create a partial function of calc_P_grad_1fea, with only one free argument
    # Split strength_grad (e by w) into a list of vectors
    term2 = torch.eye(P.shape[1],device='cuda')-(1-rst_prob)*Q.to_dense().cuda()
    # term2_inv = torch.linalg.inv(term2)
    strength_grad_split = torch.tensor_split(strength_grad, strength_grad.shape[1], dim=1)
    P_grad = np.array([calc_P_grad_1fea_cuda(edges, nnodes, M_strength, 
                          M_strength_rowSum, Q, P, rst_prob, term2, i) for i in strength_grad_split[:2]])
    return P_grad

# def calc_P_grad_cudaBatch(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, 
#                      strength_grad, ncpus=-1):
#     # # Create a partial function of calc_P_grad_1fea, with only one free argument
#     # # Split strength_grad (e by w) into a list of vectors
#     Q_grad = Q_gradient_1feature_cudaBatch(edges, nnodes, M_strength.to_dense(), M_strength_rowSum.to_dense(),
#                                 strength_grad)
#     # P_grad = iterative_P_gradient_1feature_cuda(P, Q, Q_grad, rst_prob)
#     P_grad = iterative_P_gradient_1feature_conv_cuda(P, Q, Q_grad, rst_prob, term2)
#     return P_grad

# ########################################
SRW_obj.Q, M_strength, M_strength_rowSum, strength_grad = strength_Q_and_gradient_cuda(SRW_obj.edges, SRW_obj.nnodes, SRW_obj.features, w_local)


print('Start:', time.strftime("%H:%M:%S"));P = iterative_PPR_conv_cuda_solve(SRW_obj.Q, renorm_neg_cuda(SRW_obj.P_init), SRW_obj.rst_prob);print('End:', time.strftime("%H:%M:%S"))

# print('Start:', time.strftime("%H:%M:%S"));P = iterative_PPR_conv_cuda(SRW_obj.Q, renorm_neg_cuda(SRW_obj.P_init), SRW_obj.rst_prob);print('End:', time.strftime("%H:%M:%S"))

rst_prob = SRW_obj.rst_prob
Q = SRW_obj.Q

print('Start:', time.strftime("%H:%M:%S"));P_grad = calc_P_grad_cuda(SRW_obj.edges, SRW_obj.nnodes, M_strength, M_strength_rowSum, SRW_obj.Q, P, SRW_obj.rst_prob, strength_grad);print('End:', time.strftime("%H:%M:%S"))

#########
### batch
print('Start:', time.strftime("%H:%M:%S"));term2 = torch.eye(P.shape[1])-(1-rst_prob)*Q.to_dense();P_grad_new_solve = torch.linalg.solve(term2.T,term1.T).T;print('End:', time.strftime("%H:%M:%S"))

print('Start:', time.strftime("%H:%M:%S"));term2 = torch.eye(P.shape[1])-(1-rst_prob)*Q.to_dense();term2_inv = torch.linalg.inv(term2);P_grad_new = torch.mm(term1, term2_inv);print('End:', time.strftime("%H:%M:%S"))

# strength_grad_split = torch.tensor_split(strength_grad, strength_grad.shape[1], dim=1)
# strength_grad_batch = torch.stack(strength_grad_split, dim=0)
strength_grad_batch = strength_grad.reshape(1,-1)
indices = torch.stack([edges[:, 0], edges[:, 1]])
idx = torch.arange(strength_grad.shape[1]).repeat_interleave(indices.shape[1])
indices = torch.cat((indices.repeat(1, strength_grad.shape[1]), idx.unsqueeze(0)))

M_strength_grad_batch = torch.sparse_coo_tensor(
        indices=indices,
        values=torch.squeeze(strength_grad_batch),
        size=(nnodes, nnodes, strength_grad.shape[1]))


# indices = torch.stack([edges[:, 0], edges[:, 1]])
# values_batch = strength_grad_batch.squeeze()  # Assumiamo che strength_grad_batch abbia shape [n, num_edges, 1]
# M_strength_grad_batch = torch.sparse_coo_tensor(indices.repeat(1, len(strength_grad_batch)), 
#                                                 values_batch.flatten(),
#                                                 size=(nnodes, nnodes))


# Calcolo di M_strength_grad_rowSum per il batch
M_strength_grad_rowSum_batch = torch.sparse.sum(M_strength_grad_batch, dim=1)

print('Start:', time.strftime("%H:%M:%S"))
Q_grad = (M_strength_grad_batch.to_dense().T.multiply(M_strength_rowSum.to_dense()) 
              -M_strength.to_dense().T.multiply(M_strength_grad_rowSum_batch.to_dense())) / torch.square(M_strength_rowSum).to_dense()
print('End:', time.strftime("%H:%M:%S"))

########




















# def iterative_P_gradient_1feature_conv_cuda(P, Q, Q_grad, rst_prob):
#     P_grad = torch.zeros(P.shape,device=device, dtype=torch.float64)
#     term1 = (1-rst_prob)*torch.mm(P,Q_grad)
#     term2 = torch.eye(P.shape[1], device=device)-(1-rst_prob)*Q.to_dense()
#     term2_inv = torch.linalg.inv(term2)
#     return torch.mm(term1, term2_inv)

# # not the pool
# # strength_grad_split = np.split(strength_grad.toarray(), strength_grad.shape[1], axis=1)


# ####
# # Q_grad = Q_gradient_1feature_cuda(edges, nnodes, M_strength, M_strength_rowSum,strength_grad_split[4])
# # P_grad = iterative_P_gradient_1feature_cuda(P, SRW_obj.Q, Q_grad, SRW_obj.rst_prob) 
print('Start:', time.strftime("%H:%M:%S"))
P_grad = calc_P_grad_cuda(SRW_obj.edges, SRW_obj.nnodes, M_strength, M_strength_rowSum, SRW_obj.Q, P, SRW_obj.rst_prob, strength_grad, SRW_obj.ncpus)
print('End:', time.strftime("%H:%M:%S"))

# strength_grad_split = torch.tensor_split(strength_grad, strength_grad.shape[1], dim=1)


print('Start:', time.strftime("%H:%M:%S")); P = iterative_PPR_conv_cuda(SRW_obj.Q, renorm_neg_cuda(SRW_obj.P_init), SRW_obj.rst_prob); print('End:', time.strftime("%H:%M:%S"))
