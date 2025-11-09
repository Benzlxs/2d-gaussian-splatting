#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusters to keep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def post_process_mesh_alpha(mesh, viewpoint_stack, alphamaps, depthmaps, alpha_threshold=0.5, min_view_ratio=0.5, depth_tolerance=0.1):
    """
    Post-process mesh by filtering vertices based on rendered alpha values and depth consistency.
    Projects mesh vertices back to camera views and removes background points.

    Args:
        mesh: o3d.TriangleMesh - input mesh to filter
        viewpoint_stack: list of camera viewpoints
        alphamaps: list of alpha maps (tensors) corresponding to each viewpoint
        depthmaps: list of depth maps (tensors) corresponding to each viewpoint
        alpha_threshold: threshold for considering a point as foreground (default: 0.5)
        min_view_ratio: minimum ratio of views where vertex should be foreground to keep it (default: 0.5)
        depth_tolerance: relative depth tolerance for consistency check (default: 0.1 = 10%)

    Returns:
        filtered o3d.TriangleMesh
    """
    import copy
    print(f"Alpha-based post processing: alpha_threshold={alpha_threshold}, min_view_ratio={min_view_ratio}, depth_tolerance={depth_tolerance}")

    mesh_0 = copy.deepcopy(mesh)
    vertices = np.asarray(mesh_0.vertices)
    n_vertices = len(vertices)

    if n_vertices == 0:
        print("Warning: mesh has no vertices")
        return mesh_0

    # Convert vertices to torch tensor
    vertices_torch = torch.from_numpy(vertices).float().cuda()

    # Track how many views consider each vertex as foreground
    foreground_count = np.zeros(n_vertices, dtype=np.int32)
    total_valid_views = np.zeros(n_vertices, dtype=np.int32)

    # Store alpha statistics for debugging
    alpha_values_list = [[] for _ in range(n_vertices)]

    print(f"Processing {n_vertices} vertices across {len(viewpoint_stack)} views...")

    for i, viewpoint_cam in tqdm(enumerate(viewpoint_stack), total=len(viewpoint_stack), desc="Alpha filtering"):
        # Get alpha and depth maps for this view
        alpha = alphamaps[i]
        depth = depthmaps[i]

        if alpha.dim() == 3:
            alpha = alpha[0]  # Remove channel dimension if present
        if depth.dim() == 3:
            depth = depth[0]  # Remove channel dimension if present

        # Project vertices to camera space
        vertices_homo = torch.cat([vertices_torch, torch.ones_like(vertices_torch[:, :1])], dim=-1)
        projected = vertices_homo @ viewpoint_cam.full_proj_transform

        # Perspective divide
        z = projected[:, -1:]
        pix_coords = projected[:, :2] / (projected[:, -1:] + 1e-7)

        # Check which vertices project into valid image region
        mask_in_view = ((pix_coords[:, 0] > -1.0) & (pix_coords[:, 0] < 1.0) &
                        (pix_coords[:, 1] > -1.0) & (pix_coords[:, 1] < 1.0) &
                        (z[:, 0] > 0))

        if mask_in_view.sum() == 0:
            continue

        # Sample alpha values at projected coordinates
        pix_coords_valid = pix_coords[mask_in_view].unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        alpha_samples = torch.nn.functional.grid_sample(
            alpha.cuda().float().unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            pix_coords_valid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze()  # [N]

        # Sample depth values at projected coordinates
        depth_samples = torch.nn.functional.grid_sample(
            depth.cuda().float().unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            pix_coords_valid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze()  # [N]

        # Get vertex depths in camera space
        vertex_depths = z[mask_in_view].squeeze()

        # Update counts
        mask_in_view_np = mask_in_view.cpu().numpy()
        total_valid_views[mask_in_view_np] += 1

        # Check depth consistency: vertex depth should be close to rendered depth
        depth_consistent = torch.abs(vertex_depths - depth_samples) < (depth_tolerance * depth_samples + 1e-3)

        # A vertex is considered foreground if:
        # 1. Alpha > threshold (visible in render)
        # 2. Depth is consistent (vertex is at the rendered surface, not behind or in front)
        # foreground_mask = ((alpha_samples > alpha_threshold) & depth_consistent).cpu().numpy()
        # foreground_mask = depth_consistent.cpu().numpy()
        foreground_mask = (alpha_samples > alpha_threshold).cpu().numpy()

        # Store alpha values for debugging
        valid_indices = np.where(mask_in_view_np)[0]
        alpha_np = alpha_samples.cpu().numpy()
        for idx, alpha_val in zip(valid_indices, alpha_np):
            alpha_values_list[idx].append(float(alpha_val))

        foreground_indices = valid_indices[foreground_mask]
        foreground_count[foreground_indices] += 1

    # Calculate foreground ratio for each vertex
    foreground_ratio = np.zeros(n_vertices, dtype=np.float32)
    valid_mask = total_valid_views > 0
    foreground_ratio[valid_mask] = foreground_count[valid_mask] / total_valid_views[valid_mask]

    # Keep vertices that are foreground in at least min_view_ratio of valid views
    vertices_to_keep = foreground_ratio >= min_view_ratio

    # Debug: print alpha statistics
    alpha_means = []
    for alpha_vals in alpha_values_list:
        if len(alpha_vals) > 0:
            alpha_means.append(np.mean(alpha_vals))
    if len(alpha_means) > 0:
        print(f"Alpha value statistics - Mean: {np.mean(alpha_means):.3f}, Std: {np.std(alpha_means):.3f}, Min: {np.min(alpha_means):.3f}, Max: {np.max(alpha_means):.3f}")

    print(f"Foreground ratio statistics - Mean: {foreground_ratio[valid_mask].mean():.3f}, Std: {foreground_ratio[valid_mask].std():.3f}")
    print(f"Vertices before alpha filtering: {n_vertices}")
    print(f"Vertices to keep: {vertices_to_keep.sum()}")
    print(f"Vertices to remove: {(~vertices_to_keep).sum()}")

    # Remove vertices and associated triangles
    triangles = np.asarray(mesh_0.triangles)

    # Create vertex mapping (old index -> new index)
    vertex_map = np.full(n_vertices, -1, dtype=np.int32)
    vertex_map[vertices_to_keep] = np.arange(vertices_to_keep.sum())

    # Keep only triangles where all vertices are kept
    triangles_to_keep = vertices_to_keep[triangles].all(axis=1)
    new_triangles = vertex_map[triangles[triangles_to_keep]]

    # Create new mesh
    new_vertices = vertices[vertices_to_keep]
    mesh_filtered = o3d.geometry.TriangleMesh()
    mesh_filtered.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh_filtered.triangles = o3d.utility.Vector3iVector(new_triangles)

    # Copy vertex colors if they exist
    if mesh_0.has_vertex_colors():
        vertex_colors = np.asarray(mesh_0.vertex_colors)
        mesh_filtered.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[vertices_to_keep])

    # Clean up
    mesh_filtered.remove_degenerate_triangles()
    mesh_filtered.remove_unreferenced_vertices()

    print(f"Final vertex count: {len(mesh_filtered.vertices)}")
    print(f"Final triangle count: {len(mesh_filtered.triangles)}")

    return mesh_filtered

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(),
            fx = intrins[0,0].item(),
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']*viewpoint_cam.gt_alpha_mask
            # print("Mash sum:", viewpoint_cam.gt_alpha_mask.sum())
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.alphamaps.append(gt_alpha_mask.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())

        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0.
                depth = depth.to(torch.float32)

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
