import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsearchsorted import searchsorted

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import commentjson as json
import tinycudann as tcnn

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            #print('\noutdim: ', out_dim)
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                #print('\noutdim: ', out_dim)
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    #print('\ncreating embedder')
    if i == -1:
        return nn.Identity(), input_dims
    if i == 1 and input_dims > 1:
        return 
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def get_tcnn_embedding():
    with open("configs/_config_tcnn.json") as f:
        config=json.load(f)
        
    embed_fn = tcnn.Encoding(3, config["frequency_encoding"])
    input_ch = embed_fn.n_output_dims
    embeddirs_fn = tcnn.Encoding(3, config["sh_encoding"])
    input_ch_views = embeddirs_fn.n_output_dims
    embedtime_fn = tcnn.Encoding(1, config["frequency_encoding"])
    input_ch_time = embedtime_fn.n_output_dims + 1
    
    return embed_fn, embeddirs_fn, embedtime_fn, input_ch, input_ch_views, input_ch_time
    

# TCNN Model
class FastTemporalNerf(nn.Module):
    def __init__(self, input_ch_pts=3, input_ch_view=3, input_ch_time=1,
                 output_ch_dx=3, output_ch_density=16, output_ch_color=3,
                 lrate=1e-2, zero_canonical=True,
                 debug=0):
        super(FastTemporalNerf, self).__init__()
        # Read tcnn config file
        with open("configs/_config_tcnn.json") as f:
            self.config=json.load(f)
        self.input_ch_pts=input_ch_pts
        self.input_ch_view=input_ch_view
        self.input_ch_time=input_ch_time
        self.output_ch_dx=output_ch_dx
        self.output_ch_density=output_ch_density
        self.output_ch_color=output_ch_color
        self.lrate=lrate
        self.zero_canonical=zero_canonical
        self.debug=debug
        self._is_initialized=False
        self.dx_net, self.density_net, self.rgb_net, self.pts_encode, self.t_encode = self.create_grid_model()
    
    # Create model with dx_-, density_-, and color_net aswell as (trainable) encoding
    def create_grid_model(self):
        pts_encode = tcnn.Encoding(3, self.config["grid_encoding"])
        t_encode = tcnn.Encoding(1, self.config["frequency_encoding_4"])
        dx_net = tcnn.Network(pts_encode.n_output_dims + t_encode.n_output_dims + 1, pts_encode.n_output_dims, self.config["cutlass_one"]) 
        density_net = tcnn.Network(pts_encode.n_output_dims, 16, self.config["cutlass_one"]) 
        rgb_net = tcnn.NetworkWithInputEncoding(density_net.n_output_dims + 3, 3, self.config["sh_encoding_c"], self.config["cutlass_two"])

        self._is_initialized = True
        return dx_net, density_net, rgb_net, pts_encode, t_encode

    # Return params for all networks and encoding (if trainable)
    def get_model_params(self):
        assert self._is_initialized is True, 'Model has not been initialized.'
        return [{'params': self.dx_net.parameters(), 'weight_decay': 1e-6},
                {'params': self.density_net.parameters(), 'weight_decay': 1e-6},
                {'params': self.rgb_net.parameters(), 'weight_decay': 1e-6},
                {'params': self.pts_encode.parameters()}]
        
    # Return optimizer for all networks and encoding (if trainable)
    def get_optimizer(self):
        assert self._is_initialized is True, 'Model has not been initialized.'
        return torch.optim.Adam([{'params': self.dx_net.parameters(), 'weight_decay': 1e-6},
                                 {'params': self.density_net.parameters(), 'weight_decay': 1e-6},
                                 {'params': self.rgb_net.parameters(), 'weight_decay': 1e-6},
                                 {'params': self.pts_encode.parameters()}
                                ], lr=self.lrate, eps=1e-15, betas=(0.9, 0.999))
    
    # Forward propagation
    def forward(self, x, t):
        # Split concatenated inputs x back to pts and views
        input_pts, input_views = torch.split(x, [self.input_ch_pts,self.input_ch_view], dim=-1)
        assert len(torch.unique(t)) == 1, "Only accepts all points from same time"
        
        if self.debug > 2:
            # Save tensors to file in ./test/
            i_pts = input_pts.cpu()
            i_views = input_views.cpu()
            i_time = t.cpu()
            np.savetxt('./test/input_pts.txt', i_pts.numpy())
            np.savetxt('./test/input_views.txt', i_views.numpy())
            np.savetxt('./test/time.txt', i_time.numpy())
        
        ### TODO: check if using dx_net 
        # in all cases is possible/effective
        
        input_pts_encoded = self.pts_encode(input_pts)
        cur_time = t[0, 0]
        if cur_time == 0. and self.zero_canonical:
            # No positional delta at t = 0
            # if canonical space is also at t = 0
            dx_out = torch.zeros_like(input_pts)
            density_in = input_pts_encoded
        else:
            # Encode time input and concatenate to original time input
            time_encoded = torch.cat([t, self.t_encode(t)], dim=-1)
            # Concatenate encoded input pts with encoded time
            dx_in = torch.cat([input_pts_encoded, time_encoded], dim=-1)
            # Use dx_net
            dx_out = self.dx_net(dx_in)
            density_in = input_pts_encoded + dx_out
        # Add positional delta dx (vector) to pts
        #density_in = input_pts_encoded + dx_out
        # Use density_net
        density_out = self.density_net(density_in)
        # Concatenate density_net output (16) with views (3) 
        rgb_in = torch.cat([density_out, input_views], dim=-1)
        # Use color_net (19in, 3out) internally encoded to view(16) + density(16) = 32in
        rgb_out = self.rgb_net(rgb_in)
        # Concatenate color_out with first output value of density_net
        rgba = torch.cat([rgb_out, density_out[...,:1]], dim=-1)
        
        if self.debug > 1:
            # Log tensor shapes
            print('\nforward()',
                  '\ninput_views shape: ', input_views.shape)
            if cur_time != 0. and self.zero_canonical:
                print('dx in shape: ', dx_in.shape)
            print('density in shape: ', density_in.shape,
                  '\nrgb in shape: ', rgb_in.shape)
            
            if self.debug > 2:
            # Save tensors to file in ./test/
                if cur_time != 0. and self.zero_canonical:
                    i_dx = dx_in.cpu()
                    np.savetxt('./test/dx_in.txt', i_dx.numpy())
                    o_dx = dx_out.cpu()
                    np.savetxt('./test/dx_out.txt', o_dx.numpy())
                i_dense = density_in.cpu()
                np.savetxt('./test/density_in.txt', i_dense.numpy())
                o_dense = density_out.cpu()
                np.savetxt('./test/density_out.txt', o_dense.numpy())
                i_rgb = rgb_in.cpu()
                np.savetxt('./test/rgb_in.txt', i_rgb.numpy())
                o_rgb = rgb_out.cpu()
                np.savetxt('./test/rgb_out.txt', o_rgb.numpy())
                o_rgba = rgba.cpu()
                np.savetxt('./test/rgba_out.txt', o_rgba.numpy())
        
        print(dx_out.shape)
        input()
        return rgba,dx_out

    __call__ = forward

# Model
class DirectTemporalNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = [0]
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = None
        self.zero_canonical = zero_canonical
        self._occ = NeRFOriginal(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views,
                                 input_ch_time=input_ch_time, output_ch=output_ch, skips=[0],
                                 use_viewdirs=use_viewdirs, memory=memory, embed_fn=embed_fn, output_color_ch=3)
        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 64)

    def query_time(self, pts, t, net, net_final):
        h = torch.cat([pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts, h], -1)

        return net_final(h)

    def forward(self, x, t):
        #print('first call: ', x.shape)
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = t[0, 0]
        if cur_time == 0. and self.zero_canonical:
            dx = torch.zeros_like(input_pts[:, :3])
        else:
            dx = self.query_time(input_pts, t, self._time, self._time_out)
            input_pts += dx
            #input_pts_dx = input_pts[:, :3] + dx
            #input_pts = torch.cat([input_pts_dx, self.embed_fn(input_pts_dx)], dim=-1)
        #out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)
        out, _ = self._occ(x, t)
        return out, dx
    
    def add_embedding(self, embed_fn):
        self.embed_fn = embed_fn

class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("NeRF type selected: %s" % type)

        if type == "original":
            model = NeRFOriginal(*args, **kwargs)
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
        elif type == "fast_temporal":
            model = FastTemporalNerf(*args,**kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model

class NeRFOriginal(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, output_color_ch=3, zero_canonical=True):
        super(NeRFOriginal, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] +
        #     [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs, torch.zeros_like(input_pts[:, :3])

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


def hsv_to_rgb(h, s, v):
    '''
    h,s,v in range [0,1]
    '''
    hi = torch.floor(h * 6)
    f = h * 6. - hi
    p = v * (1. - s)
    q = v * (1. - f * s)
    t = v * (1. - (1. - f) * s)

    rgb = torch.cat([hi, hi, hi], -1) % 6
    rgb[rgb == 0] = torch.cat((v, t, p), -1)[rgb == 0]
    rgb[rgb == 1] = torch.cat((q, v, p), -1)[rgb == 1]
    rgb[rgb == 2] = torch.cat((p, v, t), -1)[rgb == 2]
    rgb[rgb == 3] = torch.cat((p, q, v), -1)[rgb == 3]
    rgb[rgb == 4] = torch.cat((t, p, v), -1)[rgb == 4]
    rgb[rgb == 5] = torch.cat((v, p, q), -1)[rgb == 5]
    return rgb


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if n == 'pts_encode.params':
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is not None:
                    print(p.grad.abs().max())
                    print(p.grad.abs().mean())
                    ave_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())
                else:
                    print('null')
                    ave_grads.append(0.)
                    max_grads.append(0.)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = 0., top=max_grads[0]) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    plt.show()