from __future__ import print_function
import torch
import torch.cuda.comm as comm
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from torch.autograd import Variable
from nested_dict import nested_dict
from collections import OrderedDict
import numpy as np


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return cast(kaiming_normal(torch.Tensor(no, ni, k, k)))

def conv_params_pdpt(ni, no, k):
    w = conv_params(ni, no, k)
    if ni != no:
        return w
    else:
        n = no
        r = int(np.ceil(float(n)/2))
        slices = w.view(n, n, -1).permute(2, 0, 1)
        es, Vs = [], []
        for u in slices:
            #u_sym = u.triu(diagonal=1).t() + u.triu()
            u_sym = (u + u.t()).div(2)
            e, V = torch.symeig(u_sym)
            es.append(e[r:-1])
            Vs.append(V[:,r:-1])
        return {'e': torch.stack(es), 'V': torch.stack(Vs)}

def linear_params(ni, no):
    return cast({'weight': kaiming_normal(torch.Tensor(no, ni)), 'bias': torch.zeros(no)})


def bnparams(n):
    return cast({'weight': torch.rand(n), 'bias': torch.zeros(n)})


def bnstats(n):
    return cast({'running_mean': torch.zeros(n), 'running_var': torch.ones(n)})


def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]
    stats_replicas = [dict(zip(stats.keys(), p))
                      for p in comm.broadcast_coalesced(list(stats.values()), device_ids)]

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p, s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten_params(params):
    return OrderedDict(('.'.join(k), Variable(v, requires_grad=True))
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)


def flatten_stats(stats):
    return OrderedDict(('.'.join(k), v)
                       for k, v in nested_dict(stats).iteritems_flat())


def batch_norm(x, params, stats, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=stats[base + '.running_mean'],
                        running_var=stats[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3),
              str(tuple(v.size())).ljust(23), torch.typename(v.data if isinstance(v, Variable) else v))


def triu_diag_triut(w):
    n_output, n_input, kh, hw = w.size()
    assert n_output == n_input
    n = n_output
    # make a list of tensors of shape NxN
    tensors = map(torch.squeeze, w.view(n, n, -1).split(1, dim=2))
    # apply triu+diag+triut parameterization to every tensor and stack back
    w_hat = torch.cat([(u.triu(diagonal=1).t() + u.triu()).unsqueeze(2) for u in tensors], dim=2)
    return w_hat.view_as(w)

def triu_diag_triut_spatial(w):
    n_output, n_input, kh, kw = w.size()
    assert kh == kw
    ss = kh
    # make a list of tensors of shape 3x3
    tensors = map(torch.squeeze, w.view(-1, ss, ss).split(1, dim=0))
    # apply triu+diag+triut parameterization to every tensor and stack back
    w_hat = torch.cat([(u.triu(diagonal=1).t() + u.triu()).unsqueeze(0) for u in tensors], dim=0)
    return w_hat.view_as(w)

def symm_quarter(w):
    n_output, n_input, kh, hw = w.size()
    assert n_output == n_input
    n = n_output
    # make a list of tensors of shape NxN
    tensors = map(torch.squeeze, w.view(n, n, -1).split(1, dim=2))

    def extract_quarter(u):
        return fliplr_2d(u.triu(0)).triu(0)

    def copy_from_quarter(u):
        half = fliplr_2d(u.triu(1) + u.t())
        return half.triu(1) + half.t()

    # apply symm to every tensor and stack back
    w_hat = torch.cat([copy_from_quarter(extract_quarter(u)).unsqueeze(2) for u in tensors], dim=2)
    return w_hat.view_as(w)

def symm_one_eighth(w):
    n_output, n_input, kh, kw = w.size()
    assert n_output == n_input
    n = n_output
    # make a list of tensors of shape NxN
    tensors = map(torch.squeeze, w.view(n, n, -1).split(1, dim=2))

    def extract_quarter(u):
        u = fliplr_2d(u.triu(0)).triu(0)
        u_right = right_2d(u)
        u = u_right + fliplr_2d(u_right) # quarter itself is symmetric
        return u

    def copy_from_quarter(u):
        half = fliplr_2d(u.triu(1) + u.t())
        return half.triu(1) + half.t()

    # apply symm to every tensor and stack back
    w_hat = torch.cat([copy_from_quarter(extract_quarter(u)).unsqueeze(2) for u in tensors], dim=2)
    return w_hat.view_as(w)

def symm_block_4th(w):
    n_output, n_input, kh, kw = w.size()
    assert n_output == n_input
    n = n_output
    # make a list of tensors of shape NxN
    tensors = map(torch.squeeze, w.view(n, n, -1).split(1, dim=2))

    def extract_block(u):
        return bottom_2d( right_2d(u) )

    def copy_from_block(u):
        half = u + fliplr_2d(u)
        return half + flipud_2d(half)

    # apply symm to every tensor and stack back
    w_hat = torch.cat([copy_from_block(extract_block(u)).unsqueeze(2) for u in tensors], dim=2)
    return w_hat.view_as(w)

def w_w_transpose(w):
    return (w + w.transpose(0,1)).div(2)

def eig_decomp_reconstruct(w):
    n_output, n_input, kh, hw = w.size()
    assert n_output == n_input
    n = n_output
    # make a list of tensors of shape NxN
    tensors = map(torch.squeeze, w.view(n, n, -1).split(1, dim=2))

    def evd(x):
        e, V = torch.symeig(x.data, eigenvectors=True) # only use triu(x)
        x_hat = torch.mm(torch.mm(V, torch.diag(e)), V.t())
        return Variable(x_hat)

    # apply hat(u) = V diag(e) V' to every tensor u and stack back
    w_hat = torch.cat([evd(u).unsqueeze(2) for u in tensors], dim=2)
    return w_hat.view_as(w)

def fliplr_2d(x):
    kh, kw = x.size()
    return torch.index_select(x, dim=1, index=torch.arange(kw-1,-1,-1).long())

def flipud_2d(x):
    kh, kw = x.size()
    return torch.index_select(x, dim=0, index=torch.arange(kh-1,-1,-1).long())

def right_2d(x, center_included=True):
    kh, kw = x.size()
    n_cols = kw/2 + 1 if kw % 2 != 0 and not center_included else kw/2
    x_right = x.index_fill(dim=1, index=torch.arange(0,n_cols).long(), value=0.0)
    return x_right

def bottom_2d(x, center_included=True):
    kh, kw = x.size()
    n_rows = kh/2 + 1 if kh % 2 != 0 and not center_included else kh/2
    x_bott = x.index_fill(dim=0, index=torch.arange(0,n_rows).long(), value=0.0)
    return x_bott

