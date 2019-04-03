import torch
import torch.nn.functional as F
from .utils import conv_params, linear_params, bnparams, bnstats, \
        flatten_params, flatten_stats, batch_norm, \
        triu_diag_triut, triu_diag_triut_spatial, eig_decomp_reconstruct, w_w_transpose, \
        symm_quarter, symm_one_eighth, symm_block_4th, conv_params_pdpt

def resnet(depth, width, num_classes, symm_type='None'):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = torch.Tensor([16, 32, 64]).mul(width).int()

    def gen_block_params(ni, no):
        if symm_type == 'pdpt':
            conv1_params = conv_params_pdpt
        else:
            conv1_params = conv_params
        return {
            'conv0': conv_params(ni, no // 4),
            'conv1': conv1_params(no // 4, no // 4, k=3),
            'conv2': conv_params(no // 4, no),
            'bn0': bnparams(ni),
            'bn1': bnparams(no // 4),
            'bn2': bnparams(no // 4),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no // 4), 'bn2': bnstats(no // 4)}
                for i in range(count)}

    flat_params = flatten_params({
        'conv0': conv_params(3,16,3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': bnparams(widths[2]),
        'fc': linear_params(widths[2], num_classes),
    })

    flat_stats = flatten_stats({
        'group0': gen_group_stats(16, widths[0], n),
        'group1': gen_group_stats(widths[0], widths[1], n),
        'group2': gen_group_stats(widths[1], widths[2], n),
        'bn': bnstats(widths[2]),
    })

    def block(x, params, stats, base, mode, stride):
        if symm_type == 'tri':
            g = triu_diag_triut
        elif symm_type == 'evd':
            g = eig_decomp_reconstruct
        elif symm_type == 'spa':
            g = triu_diag_triut_spatial
        elif symm_type == 'tri+spa':
            g = lambda y: triu_diag_triut_spatial( triu_diag_triut(y) )
        elif symm_type == 'wwt':
            g = w_w_transpose
        elif symm_type == 'quar':
            g = symm_quarter
        elif symm_type == 'eight':
            g = symm_one_eighth
        elif symm_type == 'block4':
            g = symm_block_4th
        elif symm_type.startswith('chunk'):
            _, dim, n_c = symm_type.split('_')
            dim = long(dim)
            n_c = long(n_c)
            #g = lambda w: torch.cat([w.narrow(dim, 0, w.size(dim) // n_c)] * n_c, dim=dim)
            def chunk_d(w):
                n_d = w.size(dim)
                n_cc = n_d if n_c > n_d else n_c
                #assert(n_d % n_c == 0)
                return torch.cat([w.narrow(dim, 0, n_d // n_cc)] * n_cc, dim=dim)
            g = chunk_d
        elif symm_type == 'pdpt':
            def pdpt(w):
                e,V = w
                es = torch.stack(list(map(torch.diag, e)))
                #w = V.transpose(1,2) @ es @ V
                w = torch.bmm( torch.bmm(V, es), V.transpose(1,2) )
                w_hat = w.permute(1,2,0).contiguous().view(w.size(1), w.size(2), 3, 3)
                return w_hat
            g = pdpt
        else:
            g = lambda y: y
        o0 = F.relu(batch_norm(x, params, stats, base + '.bn0', mode), inplace=True)
        y0 = F.conv2d(o0, params[base + '.conv0'])
        o1 = F.relu(batch_norm(y0, params, stats, base + '.bn1', mode), inplace=True)
        if symm_type == 'pdpt':
            w_conv1 = g((params[base + '.conv1.e'], params[base + '.conv1.V']))
        else:
            w_conv1 = g(params[base + '.conv1'])
        y1 = F.conv2d(o1, w_conv1, stride=stride, padding=1)
        o2 = F.relu(batch_norm(y1, params, stats, base + '.bn2', mode), inplace=True)
        y2 = F.conv2d(o2, params[base + '.conv2'])
        if base + '.convdim' in params:
            return y2 + F.conv2d(o0, params[base + '.convdim'], stride=stride)
        else:
            return y2 + x

    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, stats, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = F.relu(batch_norm(g2, params, stats, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params, flat_stats
