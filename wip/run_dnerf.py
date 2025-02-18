import os
import imageio
import time
from numpy import Infinity
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import sys
from matplotlib import projections, pyplot as plt


from run_dnerf_helpers import *

from load_blender import load_blender_data

try:
    from apex import amp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = 0

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed)
    embedtime_fn, input_ch_time = get_embedder(args.multires, 1, args.i_embed)
    
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)
    
    # Create original or direct temporal NeRF (args.nerf_type)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                 zero_canonical=not args.not_zero_canonical).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.use_two_models_for_fine:
        model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                          zero_canonical=not args.not_zero_canonical).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn : run_network(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtime_fn=embedtime_fn,
                                                                netchunk=args.netchunk,
                                                                embd_time_discr=args.nerf_type!="temporal")

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    loss = nn.MSELoss()

    #region Half precision
    if args.do_half_precision:
        print("Run model at half precision")
        if model_fine is not None:
            [model, model_fine], optimizers = amp.initialize([model, model_fine], optimizer, opt_level='O1')
        else:
            model, optimizers = amp.initialize(model, optimizer, opt_level='O1')
    #endregion

    start = 0
    basedir = args.basedir
    expname = args.expname

    #region checkpoints
    # Load checkpoints
    if args.use_ckpts:
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            if args.do_half_precision:
                amp.load_state_dict(ckpt['amp'])
    #endregion

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine': model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : args.use_two_models_for_fine,
    }

    #region NDC LLFF
    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    #endregion

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, loss

def create_tcnn_nerf(args):
    # Initialize FastTemporalNerf Model
    print('\nDebug level: ', DEBUG)
    model = NeRF.get_by_name(args.nerf_type, lrate=args.lrate, zero_canonical=not args.not_zero_canonical, debug=DEBUG)
    assert type(model) is FastTemporalNeRF, "Wrong nerf type in args."

    ### Adam optimizer
    # Optionally get model parameters including L2 regularization on networks
    # grad_vars = model.get_model_params()
    # grad_vars = model.parameters()
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, eps=1e-15, betas=(0.9,0.999))\
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9,0.999))
    optimizer = model.get_optimizer()
    if DEBUG > 0:
        print(optimizer.param_groups)

    ### Optionally use torch L2-loss
    #loss = None
    loss = nn.MSELoss()
    
    # Build network shortcut lambda function
    # really only shortcuts passing of netchunk size, but is consistent with original implementation
    network_query_fn = lambda inputs, viewdirs, ts, network_fn : run_tcnn_network(inputs, viewdirs, ts, network_fn,
                                                                                  netchunk=args.netchunk)

    start = 0
    
    # Combine args in easy to pass dictionary
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : False,
    }
    
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['N_importance'] = 128
    render_kwargs_test['N_samples'] = 64

    return render_kwargs_train, render_kwargs_test, start, optimizer, model, loss

def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=256*64,
                embd_time_discr=True):
    """Prepares inputs and applies network 'fn'.
    inputs: N_rays x N_points_per_ray x 3
    viewdirs: N_rays x 3
    frame_time: N_rays x 1
    """
    assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    cur_time = torch.unique(frame_time)[0]

    # embed position
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    # embed time
    if embd_time_discr:
        B, N, _ = inputs.shape
        input_frame_time = frame_time[:, None].expand([B, N, 1])
        input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])
        embedded_time = embedtime_fn(input_frame_time_flat)
    else:
        assert NotImplementedError

    # embed views
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, embedded_time)
    if DEBUG > 1:
        print('\nrun_network()\nembedded shape: ', embedded.shape,
              '\nembedded_time shape: ', embedded_time.shape,
              '\noutputs_flat shape: ', outputs_flat.shape,
              '\nposition_delta_flat shape: ', position_delta_flat.shape)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta

def run_tcnn_network(inputs, viewdirs, frame_time, fn, netchunk=256*64):
    #assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    
    # Flatten inputs from (N_rays,Samples,Coords) -> (Indice,Coords)
    input_pts_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    input_views_temp = viewdirs[:,None].expand(inputs.shape)
    input_views_flat = torch.reshape(input_views_temp, [-1, input_views_temp.shape[-1]])
    # Pts and View inputs are concatenated, then passed to networks
    inputs_flat = torch.cat([input_pts_flat, input_views_flat], -1)
    
    # Flatten frame times from (N_rays,Samples,Time) -> (Indice,Time)
    B, N, _ = inputs.shape
    input_frame_time = frame_time[:, None].expand([B,N,1])
    input_frame_time_flat = torch.reshape(input_frame_time, [-1,1])
    input_frame_times_flat = torch.cat([input_frame_time_flat, input_frame_time_flat], -1)
    
    # Batchified input to models
    # get rgba output and position delta
    outputs_flat, position_delta_flat = batchify(fn, netchunk)(inputs_flat, input_frame_times_flat)
    if DEBUG > 1:
        print('\nrun_tcnn_network()\ninputs_flat shape: ', inputs_flat.shape,
              '\ninput_frame_time_flat shape: ', input_frame_time_flat.shape,
              '\noutputs_flat shape: ', outputs_flat.shape,
              '\nposition_delta_flat shape: ', position_delta_flat.shape)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(pts, frame_time):
        num_batches = pts.shape[0]
        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            out,dx = fn(pts[i:i+chunk], frame_time[i:i+chunk])
            out_list += [out]
            dx_list += [dx]
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)
    return ret

def batchify_rays(rays_flat, chunk=256*64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    Usually 500 rays with 12 fields
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # calculate alpha back from log space with torch.exp
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    #print(dists.shape)

    #sigmoid activation of rgb output
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)
    
    return rgb_map, disp_map, acc_map, weights, depth_map

def render_path(render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)

    rgbs = []
    disps = []

    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        if savedir is not None:
            rgb8_estim = to8b(rgbs[-1])
            filename = os.path.join(save_dir_estim, '{:03d}.png'.format(i+i_offset))
            imageio.imwrite(filename, rgb8_estim)
            if save_also_gt:
                rgb8_gt = to8b(gt_imgs[i])
                filename = os.path.join(save_dir_gt, '{:03d}.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_gt)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp: # lindisp is not
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand
    else:
        print('zvals was not none.')

    # Get N_samples pts along ray (eg. 64)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    
    # region plot pts
    
    #print(rays_d)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # xs0 = rays_o[...,0].cpu()
    # # ys0 = rays_o[...,1].cpu()
    # # zs0 = rays_o[...,2].cpu()
    # # xs = np.concatenate([xs0, (xs0 + rays_d[...,0].cpu())], 0)
    # # ys = np.concatenate([ys0, (ys0 + rays_d[...,1].cpu())], 0)
    # # zs = np.concatenate([zs0, (zs0 + rays_d[...,2].cpu())], 0)
    # xs = pts[...,0].cpu()
    # print(pts[...,0].shape)
    # ys = pts[...,1].cpu()
    # zs = pts[...,2].cpu()
    # ax.scatter(xs, ys, zs, marker='x', color='black')
    # for i in range(rays_o.shape[0]):
    #     ax.arrow3D(xs[i,0],
    #                ys[i,0],
    #                zs[i,0],
    #                rays_d[i,0].cpu()*4,
    #                rays_d[i,1].cpu()*4,
    #                rays_d[i,2].cpu()*4,
    #                mutation_scale=10, arrowstyle="-|>", linestyle='-', color='black')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax.set_zlim(0,1)
    # plt.show()
    
    # endregion
    
    if N_importance > 0: # Two sampling passes (N_samples then N_samples + N_importance)
        if use_two_models_for_fine:
            raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        else:
            with torch.no_grad():
                raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn)
                _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        
    #run_fn = network_fn if network_fine is None else network_fine

    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    #region show image
    
    # if do:
    #     rgb = rgb_map.reshape(40,400,3).cpu()
    #     print(rgb)
    #     plt.imshow(rgb)

    # endregion

    if DEBUG > 1:
        print('\nrender_rays()\npts shape: ', pts.shape,
              '\nviewdirs shape: ', viewdirs.shape,
              '\nframe_time shape: ', frame_time.shape,
              '\nraw shape: ', raw.shape,
              '\nposition_delta shape: ', position_delta.shape,
              '\nrgb_map shape: ', rgb_map.shape,
              '\nweights shape: ', weights.shape)

    # Returning results as dictionary
    # easy to extend (rgb0, disp0, etc.)
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals,
           'position_delta' : position_delta}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if DEBUG > 1:
        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def render(H, W, focal, chunk=256*64, rays=None, c2w=None, ndc=True,
                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  norm_min=[0.,0.,0.], norm_max=[1.,1.,1.], normalize=False,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        frame_time = torch.empty(H*W).fill_(frame_time)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        
        # normalizing rays to unit cube
        if normalize:
            rays_o_temp = torch.zeros_like(rays_o)
            rays_d_temp = torch.zeros_like(rays_d)
            
            for i in range(3):
                norm_min_flat = norm_min[i] * torch.ones_like(rays_o[...,i])
                rays_o_temp[...,i] = rays_o[...,i] - norm_min_flat
                rays_o_temp[...,i] /= norm_max[i]
                rays_d_temp[...,i] = rays_d[...,i] / norm_max[i]
                
            rays_o = rays_o_temp
            rays_d = rays_d_temp
        # region rays figure
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # xs0 = rays_o[...,0].cpu()
        # ys0 = rays_o[...,1].cpu()
        # zs0 = rays_o[...,2].cpu()
        # xs = np.concatenate([xs0, (xs0 + rays_d[...,0].cpu())], 0)
        # ys = np.concatenate([ys0, (ys0 + rays_d[...,1].cpu())], 0)
        # zs = np.concatenate([zs0, (zs0 + rays_d[...,2].cpu())], 0)
        # ax.scatter(xs, ys, zs, marker='x')
        # plt.show()
        
        # endregion
    
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None: # not happening
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    #if ndc:
        # for forward facing scenes
        #rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    
    frame_time = torch.reshape(frame_time, (frame_time.shape[0], 1))
    #frame_time = frame_time * torch.ones_like(rays_d[...,:1])

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    if DEBUG > 1:
        print('\nrender()\nrays shape: ', rays.shape)
    
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type')
    parser.add_argument("--N_iter", type=int, default=1000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=500, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--do_half_precision", action='store_true',
                        help='do half precision training and inference')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=256*64, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=256*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--use_ckpts", action='store_true',
                        help='use checkpoints')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--not_zero_canonical", action='store_true',
                        help='if set zero time is not the canonic space')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--normalize", action='store_true',
                        help='normalize input pts')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none, 1 for tcnn')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='use two models for fine results')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_iters_time", type=int, default=0,
                        help='number of steps to train on central time')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--successive_training_set", action="store_true",
                        help='train images in strict succession')
    parser.add_argument("--training_image_frequency", type=int, default=1,
                        help='number of training steps per image when training images in strict succession')
    parser.add_argument("--add_tv_loss", action='store_true',
                        help='evaluate tv loss')
    parser.add_argument("--tv_loss_weight", type=float,
                        default=1.e-4, help='weight of tv loss')
    parser.add_argument("--testset_size", type=int, default=3,
                        help='number of training images to use')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=2,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=50,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000,
                        help='frequency of render_poses video saving')

    return parser


def train(): # python3 run_dnerf.py --config configs/config.txt

    parser = config_parser()
    args = parser.parse_args()

    ### Load data ###
    if args.dataset_type == 'blender':
        images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        i_train, i_val, i_test = i_split
        
        if DEBUG > 0: 
            print('\nLoaded blender data:',
                  '\nimages: ', type(images), images.shape,
                  '\nposes: ', type(poses), poses.shape,
                  '\ntimes: ', type(times), times.shape,
                  '\nrender_poses: ', type(render_poses), render_poses.shape,
                  '\nrender_times: ', type(render_times), render_times.shape,
                  '\nheight: ', hwf[0], ' width: ', hwf[1], ' focal: ', hwf[2],
                  '\ni_train: ', i_split[0].shape,
                  ' i_val: ', i_split[1].shape,
                  ' i_test: ', i_split[2].shape,'\n')
        
        # near and far plane
        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        # images = [rgb2hsv(img) for img in images]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    #min_time, max_time = times[i_train[0]], times[i_train[-1]]
    #assert min_time == 0., "time must start at 0"
    #assert max_time == 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_times = np.array(times[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    ### Create NeRF model
    if args.nerf_type == "fast_temporal":
        render_kwargs_train, render_kwargs_test, start, optimizer, model, loss = create_tcnn_nerf(args)
    else:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, loss = create_nerf(args)
    assert render_kwargs_train is not None, "Creating nerf failed."
        
    global_step = start
    if DEBUG > 0:
        print('Created NeRF model.')

    bds_dict = {
        'near' : near,
        'far' : far
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor, save_also_gt=True)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    ### Prepare raybatch tensor if batching random rays ###
    use_batching = not args.no_batching
    if use_batching or not use_batching:
        times_flat = torch.empty(H*W,3,1).fill_(times[0]).cpu()
        for i in i_train[1:]:
            tr_time = torch.empty(H*W,3,1).fill_(times[i]).cpu()
            times_flat = np.concatenate([times_flat, tr_time], 0)
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_total = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_total = np.transpose(rays_total, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_total = np.stack([rays_total[i] for i in i_train], 0) # train images only
        rays_total = np.reshape(rays_total, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_total = np.concatenate([rays_total, times_flat], -1)
        rays_total = rays_total.astype(np.float32)
        print('done')
        i_batch = 0
        
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)
    if use_batching or not use_batching:
        if args.successive_training_set:
            rays_rgb = torch.Tensor(rays_total).to(device)
            # Normalize viewdirs to unit vectors (unit sphere)
            #rays_rgb[:,1] = torch.nn.functional.normalize(rays_rgb[:,1])
            # Find normalize_factor for mapping pts to unit cube [-1;1]
            if args.normalize:
                render_kwargs_test['normalize'] = True
                norm_min_list = [0.,0.,0.]
                norm_max_list = [1.,1.,1.]
                for i in range(3):
                    norm_min = torch.min(torch.min(rays_rgb[:,0,i] + rays_rgb[:,1,i] * near),torch.min(rays_rgb[:,0,i] + rays_rgb[:,1,i] * far))
                    rays_rgb[:,0,i] -= norm_min
                    norm_max = torch.max(torch.max(rays_rgb[:,0,i] + rays_rgb[:,1,i] * near), torch.max(rays_rgb[:,0,i] + rays_rgb[:,1,i] * far))
                    rays_rgb[:,0,i] /= norm_max
                    rays_rgb[:,1,i] /= norm_max
                    norm_min_list[i] = norm_min
                    norm_max_list[i] = norm_max
                render_kwargs_test['norm_min'] = norm_min_list
                render_kwargs_test['norm_max'] = norm_max_list
            # Start with N training examples
            rays_rgb = rays_rgb[:H*W*3]
            rays_total = torch.Tensor(rays_total[H*W*3:]).to(device)
            # Shuffle rays of first N training examples
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
        else:
            rays_rgb = torch.Tensor(rays_total).to(device)
            # Normalizing rays to unit cube [0,1] over x,y,z (one-factor)
            if args.normalize:
                render_kwargs_test['normalize'] = True
                norm_min_list = [0.,0.,0.]
                norm_max_list = [1.,1.,1.]
                for i in range(3):
                    norm_min = torch.min(torch.min(rays_rgb[:,0,i] + rays_rgb[:,1,i] * near),torch.min(rays_rgb[:,0,i] + rays_rgb[:,1,i] * far))
                    rays_rgb[:,0,i] -= norm_min
                    norm_max = torch.max(torch.max(rays_rgb[:,0,i] + rays_rgb[:,1,i] * near), torch.max(rays_rgb[:,0,i] + rays_rgb[:,1,i] * far))
                    rays_rgb[:,0,i] /= norm_max
                    rays_rgb[:,1,i] /= norm_max
                    norm_min_list[i] = norm_min
                    norm_max_list[i] = norm_max
                render_kwargs_test['norm_min'] = norm_min_list
                render_kwargs_test['norm_max'] = norm_max_list
            # Shuffle rays
            
            # region print
            # print(torch.min(torch.min(rays_rgb[:,0,0] + rays_rgb[:,1,0] * near),torch.min(rays_rgb[:,0,0] + rays_rgb[:,1,0] * far)))
            # print(torch.min(torch.min(rays_rgb[:,0,1] + rays_rgb[:,1,1] * near),torch.min(rays_rgb[:,0,1] + rays_rgb[:,1,1] * far)))
            # print(torch.min(torch.min(rays_rgb[:,0,2] + rays_rgb[:,1,2] * near),torch.min(rays_rgb[:,0,2] + rays_rgb[:,1,2] * far)))
            # print(torch.max(torch.max(rays_rgb[:,0,0] + rays_rgb[:,1,0] * near), torch.max(rays_rgb[:,0,0] + rays_rgb[:,1,0] * far)))
            # print(torch.max(torch.max(rays_rgb[:,0,1] + rays_rgb[:,1,1] * near), torch.max(rays_rgb[:,0,1] + rays_rgb[:,1,1] * far)))
            # print(torch.max(torch.max(rays_rgb[:,0,2] + rays_rgb[:,1,2] * near), torch.max(rays_rgb[:,0,2] + rays_rgb[:,1,2] * far)))
            # endregion
            
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]

    # Summary writers
    #writer = SummaryWriter(os.path.join(basedir, 'summaries', expname,
    #                                    'learn_rate', str(args.lrate)))
    
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname, 'tcnntest2', expname))
    
    ### Input no. of iterations manually or get from args ###
    #iterations = int(input('\nInput no. of iterations: '))
    #N_iters = iterations + 1
    N_iters = args.N_iter + 1
    
    # Rays per batch
    N_rand = args.N_rand
    
    
    
    ### Start of training loop ###
    print('\nBegin')  
    start = start + 1
    for i in trange(start, N_iters):
        if DEBUG > 1:
            print('\nIteration ', i)
        time0 = time.time()

        if use_batching:
            # region print training image
            ## add -> first_image = rays_rgb[:160_000] before shuffling of rays_rgb
            ## change chunk to 16000
            # if i == 4999:
                
            #     batch = first_image[:1000]
            #     batch = torch.transpose(batch, 0,1)
            #     batch_rays, frame_time = batch[:2,...,:3],batch[1,...,3]
            #     with torch.no_grad():
            #         rendered, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
            #                                     verbose=i < 10, retraw=True,
            #                                     **render_kwargs_train)

            #     i_batch=1000
            #     for y in range(159):
            #         batch = first_image[i_batch:i_batch+1000]
            #         batch = torch.transpose(batch, 0,1)
            #         batch_rays, frame_time = batch[:2,...,:3],batch[1,...,3]
            #         i_batch += 1000

            #         with torch.no_grad():
            #             rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
            #                                     verbose=i < 10, retraw=True,
            #                                     **render_kwargs_train)

            #         rendered = torch.cat([rendered, rgb], dim=0)

            #     rendered = rendered.reshape(400,400,3)
            #     plt.imshow(rendered.cpu())
            #     plt.show()
            # endregion
            
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, frame_time = batch[:2,...,:3], batch[2,...,:3], batch[1,...,3]
            
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
            
            # region successive training
            ## add psnr = 0 before loop
            
            # if args.successive_training_set:
            #     if i%10 == 0:
            #         if rays_total.shape[0] > 0:
            #             rays_rgb = torch.cat([rays_rgb, rays_total[:H*W]], 0)
            #             rays_total = rays_total[H*W:]
            #             rand_idx = torch.randperm(rays_rgb.shape[0])
            #             rays_rgb = rays_rgb[rand_idx]
            #             i_batch = 0
            # endregion
                
        else:
            # Select Training Example Random from one image
            if i >= args.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                skip_factor = i / float(args.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])
            tr_ex = img_i

            #target     = comparison for predicted RGB colors for selected training example img_i
            #pose       = transformation matrix of selected training example (first 3 quadruples)
            #frame_time = timestamp of selected training example
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            frame_time = torch.empty(N_rand).fill_(times[img_i])
            
            if N_rand is not None:

                #get H x W ray origins, and directions from height, width, focal and camera position
                #c2w (camera to world) transformation
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                
                # Get img plane coordinates -> eg. 400x400 (H x W) coords grid for same resolution training example
                # Center cropping by precrop_frac until precrop_iters <- focus early iterations on center of scene
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                             torch.meshgrid(
                             torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                             torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW),
                             indexing='ij'
                             ), -1)
                    if i == start and DEBUG > 0:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)

                # Select N_rand ray origins and directions and concatenate to 'batch_rays'
                # (2, N_rand, 3) with (type(origin/direction), indice, coordinate)
                # aswell as target RGB in 'target_s' (indice, RGB)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2) eg. 16000 * [x y]
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,) eg. 500 indices
                select_coords = coords[select_inds].long()  # (N_rand, 2) eg. 500 * [x y]
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                if args.normalize:
                    for norm_i in range(3):
                        rays_o[:,norm_i] -= norm_min_list[norm_i]
                        rays_o[:,norm_i] /= norm_max_list[norm_i]
                        rays_d[:,norm_i] /= norm_max_list[norm_i]
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                if DEBUG > 1:
                    print(f"\ntrain()\nrays_o/rays_d shape: {rays_o.shape} / {rays_d.shape}\nbatch_rays shape: {batch_rays.shape}")

        ###  Core optimization loop  ###
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        ### Loss calculation, backward propagation and optimizer step
        #region TV Loss
        tv_loss = 0.
        if args.add_tv_loss:
            frame_time_prev = times[img_i - 1] if img_i > 0 else None
            frame_time_next = times[img_i + 1] if img_i < times.shape[0] - 1 else None
            
            if frame_time_prev is not None and frame_time_next is not None:
                if np.random.rand() > .5:
                    frame_time_prev = None
                else:
                    frame_time_next = None
                    
            if frame_time_prev is not None:
                rand_time_prev = frame_time_prev + (frame_time - frame_time_prev) * torch.rand(1)[0]
                _, _, _, extras_prev = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=rand_time_prev,
                                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                **render_kwargs_train)
            if frame_time_next is not None:
                rand_time_next = frame_time + (frame_time_next - frame_time) * torch.rand(1)[0]
                _, _, _, extras_next = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=rand_time_next,
                                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                **render_kwargs_train)
                
            if frame_time_prev is not None:
                tv_loss += ((extras['position_delta'] - extras_prev['position_delta']).pow(2)).sum()
                if 'position_delta_0' in extras:
                    tv_loss += ((extras['position_delta_0'] - extras_prev['position_delta_0']).pow(2)).sum()
            if frame_time_next is not None:
                tv_loss += ((extras['position_delta'] - extras_next['position_delta']).pow(2)).sum()
                if 'position_delta_0' in extras:
                    tv_loss += ((extras['position_delta_0'] - extras_next['position_delta_0']).pow(2)).sum()
            tv_loss = tv_loss * args.tv_loss_weight
        #endregion
        
        optimizer.zero_grad(True)
        
        ## Torch vs Self-implemented L2-loss calculation
        # when using Torch version uncomment in create_nerf() / create_tcnn_nerf()
        #output = loss(rgb, target_s)
        output = img2mse(rgb, target_s)
        #output = output + tv_loss
        psnr = mse2psnr(output)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.do_half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            output.backward()
        
        #plot_grad_flow(model.named_parameters())

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # learn rate will decay by decay factor 
        # decay_factor = 0.1
        # decay_steps = args.lrate_decay #  * 1000
        # new_lrate = args.lrate * (decay_factor ** (global_step / decay_steps))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        ### Weight saving
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                #'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            #if render_kwargs_train['network_fine'] is not None:
            #    save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            if args.do_half_precision:
                save_dict['amp'] = amp.state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
            
        ### PRINT LOSS
        if i%args.i_print == 0:
            #tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {output.item()} PSNR: {psnr.item()} on IMG: {img_i}"
            tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {output.item()} PSNR: {psnr.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
            tqdm.write(tqdm_txt)

            writer.add_scalar('loss', output.item(), i)
            writer.add_scalar('psnr', psnr.item(), i)
            if 'rgb0' in extras:
                writer.add_scalar('loss0', img_loss0.item(), i)
                writer.add_scalar('psnr0', psnr0.item(), i)
            if args.add_tv_loss:
                writer.add_scalar('tv', tv_loss.item(), i)
                
            #print(optimizer.param_groups[0])

        #del output, psnr, target_s
        del output, target_s
        if 'rgb0' in extras:
            del img_loss0, psnr0
        if args.add_tv_loss:
            del tv_loss
        del rgb, disp, acc, extras

        ### TENSORBOARD IMG SAVING
        if i%args.i_img==0:
            torch.cuda.empty_cache()
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            frame_time = times[img_i]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, focal, chunk=256*64, c2w=pose, frame_time=frame_time,
                                                    **render_kwargs_test)

            #psnr = mse2psnr(img2mse(rgb, target))
            writer.add_image('gt', to8b(target.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('rgb', to8b(rgb.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('disp', disp.cpu().numpy(), i, dataformats='HW')
            writer.add_image('acc', acc.cpu().numpy(), i, dataformats='HW')

            if 'rgb0' in extras:
                writer.add_image('rgb_rough', to8b(extras['rgb0'].cpu().numpy()), i, dataformats='HWC')
            if 'disp0' in extras:
                writer.add_image('disp_rough', extras['disp0'].cpu().numpy(), i, dataformats='HW')
            if 'z_std' in extras:
                writer.add_image('acc_rough', extras['z_std'].cpu().numpy(), i, dataformats='HW')

            print("finish summary")
            writer.flush()

        if i%args.i_video==0:
            # Turn on testing mode
            print("Rendering video...")
            with torch.no_grad():
                savedir = os.path.join(basedir, expname, 'frames_{}_spiral_{:06d}_time/'.format(expname, i))
                rgbs, disps = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, savedir=savedir)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                with torch.no_grad():
                    rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        ### TESTSET
        if i%args.i_testset==0:
            #torch.cuda.empty_cache()
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            print('Testing poses shape...', poses[i_test[[4,9,16]]].shape)

            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test[[4,9,16]]]).to(device), torch.Tensor(times[i_test[[4,9,16]]]).to(device),
                            hwf, 128*32, render_kwargs_test, gt_imgs=images[i_test[[4,9,16]]], savedir=testsavedir)
            print('Saved test set')
            
        if DEBUG > 1:
            print(f"\nIteration {i} completed. Learned from training example {tr_ex}.")
            input('Press Enter to start next iteration.')

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
