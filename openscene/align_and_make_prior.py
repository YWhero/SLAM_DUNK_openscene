import argparse
import numpy as np
import torch
import open3d as o3d
import itertools

# ===== Mosaic3D 좌표계 -> refined(mesh) 좌표계 역변환 =====
# convert_stray.py 에서:
# points_mosaic = points_refined @ R.T
# 이므로 여기서는 points_refined = points_mosaic @ R
R_MOSAIC_TO_REFINED = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
], dtype=np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_ply", required=True)
    ap.add_argument("--mosaic_pth", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--max_dist", type=float, default=0.05)
    ap.add_argument("--voxel", type=float, default=0.05, help="downsample voxel for alignment")
    ap.add_argument("--icp_thresh", type=float, default=0.20, help="ICP correspondence threshold (m)")
    ap.add_argument("--label_id", type=int, required=True, help="label id to build prior from (e.g., wall=0, floor=1)")
    ap.add_argument("--sample_mesh", type=int, default=60000)
    ap.add_argument("--sample_mosaic", type=int, default=60000)
    return ap.parse_args()

def to_pcd(xyz: np.ndarray):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return p

def random_sample(xyz, n, seed=0):
    if xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx]

def nn_median_dist(src_xyz, dst_kdtree, dst_pcd):
    # compute median NN distance from src -> dst
    dists = []
    for p in src_xyz:
        _, _, dd = dst_kdtree.search_knn_vector_3d(p, 1)
        if len(dd) == 0:
            continue
        dists.append(np.sqrt(dd[0]))
    if not dists:
        return np.inf
    return float(np.median(dists))

def main():
    args = parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh_ply)
    V = np.asarray(mesh.vertices).astype(np.float32)

    d = torch.load(args.mosaic_pth, map_location="cpu")

    # Mosaic3D 좌표 (convert_stray.py에서 축 변환된 좌표)
    P = d["coord"].cpu().numpy().astype(np.float32)
    pred = d["pred"].cpu().numpy().astype(np.int32)

    # ===== 핵심: Mosaic3D coord -> refined(mesh) 좌표계로 역변환 =====
    P = P @ R_MOSAIC_TO_REFINED


    # --- build point clouds for ICP alignment (use downsample) ---
    src = to_pcd(P)
    tgt = to_pcd(V)

    src_ds = src.voxel_down_sample(args.voxel)
    tgt_ds = tgt.voxel_down_sample(args.voxel)

    # initial translation: match centers
    src_center = np.asarray(src_ds.points).mean(axis=0)
    tgt_center = np.asarray(tgt_ds.points).mean(axis=0)
    T_init = np.eye(4, dtype=np.float64)
    T_init[:3, 3] = (tgt_center - src_center)

    # ICP (point-to-point) for rigid alignment
    reg = o3d.pipelines.registration.registration_icp(
        src_ds, tgt_ds,
        args.icp_thresh,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T_icp = reg.transformation
    print("[ICP] fitness:", reg.fitness, "rmse:", reg.inlier_rmse)
    print("[ICP] T:\n", T_icp)

    # transform full mosaic points
    P_h = np.concatenate([P.astype(np.float64), np.ones((P.shape[0], 1))], axis=1)
    P_aligned = (T_icp @ P_h.T).T[:, :3].astype(np.float32)

    # --- create floor prior by NN to floor points ---
    mask = (pred == args.label_id)
    P_cls = P_aligned[mask]
    print("mosaic points:", P.shape[0], "selected label points:", P_cls.shape[0], "label_id:", args.label_id)

    cls_pcd = to_pcd(P_cls)
    cls_kdt = o3d.geometry.KDTreeFlann(cls_pcd)


    prior = np.zeros((V.shape[0],), dtype=np.uint8)
    max_d2 = args.max_dist * args.max_dist

    # iterate mesh vertices and mark if near any floor point
    hit = 0
    for i, v in enumerate(V.astype(np.float64)):
        _, _, dd = cls_kdt.search_knn_vector_3d(v, 1)
        if len(dd) == 0:
            continue
        if dd[0] <= max_d2:
            prior[i] = 1
            hit += 1
    
    prior = prior.astype(np.float32) 
    np.save(args.out_npy, prior)
    print(f"Saved prior: {args.out_npy}")
    print(f"mesh vertices: {V.shape[0]}")
    print(f"prior positives: {hit} ({100.0*hit/V.shape[0]:.2f}%)")
    print(f"max_dist: {args.max_dist} m")

if __name__ == "__main__":
    main()
