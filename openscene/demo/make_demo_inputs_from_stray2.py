import os
import numpy as np
import torch
import open3d as o3d

def main():
    # ====== INPUTS (너 경로 그대로) ======
    MESH_PLY = "/home/yongbin53/yw/openscene/data/stray_data/mesh_refined.ply"
    PCD_PLY  = "/home/yongbin53/yw/openscene/data/stray_data/pointcloud_refined.ply"
    FEAT_PT  = "/home/yongbin53/yw/openscene/stray_data_lseg/lseg_fused_feature_512/pointcloud_feat_lseg512.pt"

    # ====== OUTPUTS (demo가 읽는 폴더) ======
    DEMO_DIR = "/home/yongbin53/yw/openscene/demo"
    OUT_REGION_DIR = os.path.join(DEMO_DIR, "region_segmentations")
    OUT_FEAT_DIR   = os.path.join(DEMO_DIR, "openscene_features")
    os.makedirs(OUT_REGION_DIR, exist_ok=True)
    os.makedirs(OUT_FEAT_DIR, exist_ok=True)

    OUT_REGION_PLY = os.path.join(OUT_REGION_DIR, "stray_region0.ply")
    OUT_FEAT_NPY   = os.path.join(OUT_FEAT_DIR, "RP0_stray_region0.npy")

    # ====== LOAD MESH ======
    assert os.path.exists(MESH_PLY), MESH_PLY
    mesh = o3d.io.read_triangle_mesh(MESH_PLY)
    if not mesh.has_vertices():
        raise RuntimeError("mesh has no vertices")
    V = np.asarray(mesh.vertices).astype(np.float32)  # (Nv,3)
    Nv = V.shape[0]
    print("[mesh] vertices:", Nv, "triangles:", np.asarray(mesh.triangles).shape[0])

    # ====== LOAD POINTCLOUD ======
    assert os.path.exists(PCD_PLY), PCD_PLY
    pcd = o3d.io.read_point_cloud(PCD_PLY)
    P = np.asarray(pcd.points).astype(np.float32)  # (Np,3)
    Np = P.shape[0]
    print("[pcd] points:", Np)

    # ====== LOAD FEATURES ======
    assert os.path.exists(FEAT_PT), FEAT_PT
    d = torch.load(FEAT_PT, map_location="cpu")

    if "feat" not in d:
        raise KeyError(f"FEAT_PT must contain key 'feat'. keys={list(d.keys())}")

    F = d["feat"]
    if torch.is_tensor(F):
        F = F.detach().cpu().numpy()
    F = F.astype(np.float32)

    if F.shape[0] != Np:
        raise RuntimeError(f"feature rows != pointcloud points: feat {F.shape[0]} vs pcd {Np}")

    if F.shape[1] != 512:
        raise RuntimeError(f"expected 512-dim features, got {F.shape}")

    print("[feat] shape:", F.shape, "dtype:", F.dtype)

    # ====== BUILD NN SEARCH (mesh vertex -> nearest point) ======
    # Open3D KDTree는 "pointcloud" 대상으로 구성 가능
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    Vfeat = np.zeros((Nv, 512), dtype=np.float32)
    nn_dist2 = np.zeros((Nv,), dtype=np.float32)

    # NOTE: Nv가 크면 시간이 좀 걸릴 수 있음. 그래도 1번만 하면 됨.
    for i in range(Nv):
        query = V[i]
        k, idx, dist2 = kdtree.search_knn_vector_3d(query, 1)
        if k < 1:
            continue
        j = idx[0]
        Vfeat[i] = F[j]
        nn_dist2[i] = dist2[0]

        if (i+1) % 200000 == 0:
            print(f"  mapped {i+1}/{Nv} vertices...")

    print("[map] done. nn_dist2 stats:",
          "min", float(nn_dist2.min()),
          "mean", float(nn_dist2.mean()),
          "max", float(nn_dist2.max()))

    # ====== OPTIONAL: normalize vertex features (demo는 dot product로 쓰므로 정규화 추천) ======
    # text는 clip_server에서 normalize 하고 있으니, vertex도 normalize하면 안정적임
    denom = np.linalg.norm(Vfeat, axis=1, keepdims=True) + 1e-6
    Vfeat = Vfeat / denom

    # ====== SAVE outputs ======
    # 1) region0.ply: mesh 그대로 저장 (색이 있어도 OK)
    o3d.io.write_triangle_mesh(OUT_REGION_PLY, mesh)
    print("[save] region ply:", OUT_REGION_PLY)

    # 2) RP0_region0.npy: (Nv,512) 저장
    np.save(OUT_FEAT_NPY, Vfeat.astype(np.float32))
    print("[save] feature npy:", OUT_FEAT_NPY, "shape:", Vfeat.shape, "dtype:", Vfeat.dtype)

    print("DONE")

if __name__ == "__main__":
    main()
