import plotly.graph_objects as go
import numpy as np
import torch
from skimage import measure

def plot_mesh(vertices, triangles, title="Mesh", opacity=1.0, show_axes=False):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color='lightgray',
            opacity=opacity,
            flatshading=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                fresnel=0.5,
                specular=0.5,
                roughness=0.5
            ),
            lightposition=dict(x=100, y=200, z=150)
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def extract_mesh_from_sdf(model, device, min_bound, max_bound, center, scale, grid_res=128, batch_size=65536, padding=(0, 0, 0)):
    model.to(device)
    model.eval()

    x = np.linspace(min_bound[0] - padding[0], max_bound[0] + padding[0], grid_res)
    y = np.linspace(min_bound[1] - padding[1], max_bound[1] + padding[1], grid_res)
    z = np.linspace(min_bound[2] - padding[2], max_bound[2] + padding[2], grid_res)

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    pts_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    pts = (pts_world - center) / scale

    preds = []

    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            batch = torch.from_numpy(pts[i:i+batch_size]).to(device)
            pred = model(batch)
            preds.append(pred.cpu().numpy())

    sdf = np.vstack(preds).reshape(grid_res, grid_res, grid_res)

    verts, faces, normals, values = measure.marching_cubes(sdf, level=0)

    verts[:, 0] = (verts[:, 0] / (grid_res - 1)) * (x[-1] - x[0]) + x[0]
    verts[:, 1] = (verts[:, 1] / (grid_res - 1)) * (y[-1] - y[0]) + y[0]
    verts[:, 2] = (verts[:, 2] / (grid_res - 1)) * (z[-1] - z[0]) + z[0]

    return verts, faces
