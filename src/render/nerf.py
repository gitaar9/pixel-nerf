"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
import torch.nn.functional as F
import util
import torch.autograd.profiler as profiler
from torch.nn import DataParallel
from dotmap import DotMap


def plot_points(coordinates_list, rgb, alpha, plt_ax=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # need to plot the points using their alpha/rgb
    rgb = rgb.cpu().detach().numpy().squeeze()
    alpha = alpha.cpu().detach().numpy().squeeze()
    alpha = alpha / np.max(alpha)
    print("RGB array shape: ", rgb.shape, " Min and max: ", np.min(rgb), np.max(rgb))
    print("Alpha array shape: ", alpha.shape, " Min and max: ", np.min(alpha), np.max(alpha))
    rgba = np.concatenate((rgb, np.expand_dims(alpha, axis=1)), axis=1)  # Combine rgb and alpha

    # Filter out points with to little alpha or completely white points
    zipped = [(c, (r, g, b, a)) for c, (r, g, b, a) in list(zip(coordinates_list, rgba)) if a > 0.01 and ((b + g + r) < 2.9)]
    coordinates_list, rgba = zip(*zipped)
    print("Nr of points left after filtering: ", len(coordinates_list))

    # Plot the points in 3D
    if plt_ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    else:
        ax = plt_ax
    ax.view_init(elev=0, azim=90)

    ax.scatter(*list(zip(*coordinates_list)), marker='.', c=list(rgba))

    if plt_ax is None:
        limit = .25
        ax.set_xlim3d(-limit, limit)
        ax.set_ylim3d(-limit, limit)
        ax.set_zlim3d(-limit, limit)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if plt_ax is None:
        plt.show()


def plot_3d_points(coordinates_list, max_points=15000, extra_point=None, plt_ax=None, strange_alpha=False, alpha=1.):
    import numpy as np
    import matplotlib.pyplot as plt
    if len(coordinates_list) > max_points:
        coordinates_list = coordinates_list.copy()
        idxs = np.random.choice(np.arange(len(coordinates_list)), 15000, replace=False)
        coordinates_list = coordinates_list[idxs]
    if plt_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        ax = plt_ax

    rgba = [(.2, .2, .6, alpha) for _ in range(len(coordinates_list))]
    if strange_alpha:  # make pixels further away from mean less transparant
        c_l_np = np.asarray(coordinates_list)
        avg_coord = np.mean(c_l_np, axis=0)
        avg_coord = np.concatenate([np.expand_dims(avg_coord, 0)] * c_l_np.shape[0], axis=0)
        distance = (c_l_np - avg_coord) ** 2
        distance = np.sum(distance, axis=1)
        distance /= np.max(distance)
        distance /= 2
        distance **= 2
        rgba = [(0.2, 0.2, .6, a) for a in list(distance)]
    ax.scatter(*list(zip(*coordinates_list)), marker='.', c=rgba)
    if extra_point is not None:
        ax.scatter(*extra_point, marker='*')
    if plt_ax is None:
        plt.show()


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, want_weights=False, mirror_x=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.net, rays, want_weights=want_weights and not self.simple_output, mirror_x=mirror_x
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def invert_tensor(self, t):
        t = t.clone()
        if len(t.shape) == 2:
            t[:, 0] = t[:, 0] * -1
        if len(t.shape) == 3:
            t[:, :, 0] = t[:, :, 0] * -1
        return t

    def composite(self, model, rays, z_samp, coarse=True, sb=0, mirror_x=False):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            # My shitty thing that calculates everything twice
            single_rays = rays
            single_z_samp = z_samp

            if mirror_x:
                rays = torch.cat([rays, rays], dim=0)
                z_samp = torch.cat([z_samp, z_samp], dim=0)

            B, K = z_samp.shape

            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            #  if far:
            #      delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            # (B, K, 3)
            points = single_rays[:, None, :3] + single_z_samp.unsqueeze(2) * single_rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs

            val_all = []
            if sb > 0:
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            # My invert CODE!
            if mirror_x:
                points = torch.cat([points, self.invert_tensor(points)], dim=0)
            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if use_viewdirs:
                dim1 = K
                viewdirs = single_rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)

                # My invert CODE!
                if mirror_x:
                    viewdirs = torch.cat([viewdirs, self.invert_tensor(viewdirs)], dim=0)
                split_viewdirs = torch.split(
                    viewdirs, eval_batch_size, dim=eval_batch_dim
                )
                for pnts, dirs in zip(split_points, split_viewdirs):
                    val_all.append(model(pnts, coarse=coarse, viewdirs=dirs))
            else:
                for pnts in split_points:
                    val_all.append(model(pnts, coarse=coarse))

            points = None  # RESTORE TO DELETING THIS
            viewdirs = None
            # (B*K, 4) OR (SB, B'*K, 4)
            out = torch.cat(val_all, dim=eval_batch_dim)
            out = out.reshape(B, K, -1)  # (B, K, 4 or 5)

            rgbs = out[..., :3]  # (B, K, 3)
            sigmas = out[..., 3]  # (B, K)
            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            # compute the gradients in log space of the alphas, for NV TV occupancy regularizer
            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)

            # # Start of my bullshit
            # print(sb, B, K)
            # t = points.cpu().detach().numpy()
            # print(t.shape)
            #
            # print(t[0].shape)
            # print(rgbs.reshape(t.shape)[0].shape)
            # print(alphas.reshape(t.shape[:2])[0].shape)
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            #
            # plot_points(t[0],
            #             rgbs.reshape(t.shape)[0],
            #             alphas.reshape(t.shape[:2])[0],
            #             plt_ax=ax)
            # print(viewdirs.shape)
            # a_view_dir = viewdirs.cpu().squeeze().numpy()[0][0]
            # print("1: ", a_view_dir)
            # ax.quiver(0, 0, 0, *a_view_dir)
            # limit = .5
            # ax.set_xlim3d(-limit, limit)
            # ax.set_ylim3d(-limit, limit)
            # ax.set_zlim3d(-limit, limit)
            #
            #
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # plot_points(t[1],
            #             rgbs.reshape(t.shape)[1],
            #             alphas.reshape(t.shape[:2])[1],
            #             plt_ax=ax)
            # a_view_dir = viewdirs.cpu().squeeze().numpy()[1][0]
            # print("1: ", a_view_dir)
            # ax.quiver(0, 0, 0, *a_view_dir)
            # limit = .5
            # ax.set_xlim3d(-limit, limit)
            # ax.set_ylim3d(-limit, limit)
            # ax.set_zlim3d(-limit, limit)
            #
            # plt.show()
            #
            # # END



            deltas = None
            sigmas = None
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)
            # alphas = None
            alphas_shifted = None

            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
            depth_final = torch.sum(weights * z_samp, -1)  # (B)
            if self.white_bkgd:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
            return (
                weights[:weights.shape[0] // 2] if mirror_x else weights,
                rgb_final[:rgb_final.shape[0] // 2] if mirror_x else rgb_final,
                depth_final[:depth_final.shape[0] // 2] if mirror_x else depth_final,
                alphas[:alphas.shape[0] // 2] if mirror_x else alphas,
                alphas[alphas.shape[0] // 2:] if mirror_x else None,
            )

    def forward(
        self, model, rays, want_weights=False, mirror_x=False
    ):
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            rays = rays.reshape(-1, 8)  # (SB * B, 8)

            z_coarse = self.sample_coarse(rays)  # (B, Kc)
            coarse_composite = self.composite(
                model, rays, z_coarse, coarse=True, sb=superbatch_size, mirror_x=mirror_x
            )

            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite, superbatch_size, want_weights=want_weights
                ),
            )

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[2])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                fine_composite = self.composite(
                    model, rays, z_combine_sorted, coarse=False, sb=superbatch_size, mirror_x=mirror_x
                )
                outputs.fine = self._format_outputs(
                    fine_composite, superbatch_size, want_weights=want_weights
                )

            return outputs

    def _format_outputs(
        self, rendered_outputs, superbatch_size, want_weights=False
    ):
        weights, rgb, depth, alphas, inverse_alphas = rendered_outputs
        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
        ret_dict = DotMap(rgb=rgb, depth=depth, alphas=alphas, inverse_alphas=inverse_alphas)
        if want_weights:
            ret_dict.weights = weights
        return ret_dict

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
