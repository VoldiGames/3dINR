import numpy as np
import point_cloud_utils as pcu
import torch 

def get_fixed_sdf_dataset(mesh, num_samples=500000, noise_surface=1e-5, noise_near=5e-3):
    verts_np = np.asarray(mesh.vertices).astype(np.float32)
    faces_np = np.asarray(mesh.triangles).astype(np.int32)

    n_surface = int(num_samples * 0.7)
    n_near    = int(num_samples * 0.1)
    n_uniform = int(num_samples * 0.2)

    pcd_surface = mesh.sample_points_uniformly(n_surface)
    pts_surface = np.asarray(pcd_surface.points).astype(np.float32)
    pts_surface += np.random.normal(scale=noise_surface, size=pts_surface.shape)

    pcd_near = mesh.sample_points_uniformly(n_near)
    pts_near = np.asarray(pcd_near.points).astype(np.float32)
    pts_near += np.random.normal(scale=noise_near, size=pts_near.shape)

    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = np.asarray(bbox.min_bound).astype(np.float32)
    max_bound = np.asarray(bbox.max_bound).astype(np.float32)
    padding = 0.1 * (max_bound - min_bound)

    pts_uniform = np.random.uniform(low=min_bound - padding, high=max_bound + padding, size=(n_uniform, 3)).astype(np.float32)

    query_points = np.vstack([pts_surface, pts_near, pts_uniform])

    sdf = pcu.signed_distance_to_mesh(query_points, verts_np, faces_np)[0]

    pts_min = query_points.min(axis=0)
    pts_max = query_points.max(axis=0)

    center = ((pts_max + pts_min) / 2).astype(np.float32)
    scale = np.float32((pts_max - pts_min).max() / 2)

    query_points = (query_points - center) / scale
    sdf = sdf / scale

    return (
        torch.from_numpy(query_points).float(),
        torch.from_numpy(sdf).float().view(-1, 1),
        center,
        scale,
        min_bound,
        max_bound,
        padding
    )
