# /home/yongbin53/yw/openscene/demo/make_stray_demo_inputs.py
import os
import numpy as np
import torch
import open3d as o3d

def main():
    # ====== 너가 가진 입력 ======
    # 1) 데모에 띄울 형상 (가장 간단: pointcloud)
    PLY_IN = "/home/yongbin53/yw/openscene/data/stray_data/pointcloud_refined.ply"
    # (mesh 쓰고 싶으면 아래로 바꿔도 됨)
    # PLY_IN = "/home/yongbin53/yw/openscene/data/stray_data/mesh_refined.ply"

    # 2) OpenScene 형식으로 저장된 fused feature (mask+feat 압축)
    PT_IN = "/home/yongbin53/yw/openscene/stray_data_lseg/openscene_input_lseg512/stray_scene_0.pt"

    assert os.path.exists(PLY_IN), PLY_IN
    assert os.path.exists(PT_IN), PT_IN

    # ====== 데모 폴더 구조( run_demo가 읽는 위치 ) ======
    demo_root = os.path.dirname(os.path.abspath(__file__))
    ply_out_dir = os.path.join(demo_root, "region_segmentations")
    feat_out_dir = os.path.join(demo_root, "openscene_features")
    os.makedirs(ply_out_dir, exist_ok=True)
    os.makedirs(feat_out_dir, exist_ok=True)

    PLY_OUT = os.path.join(ply_out_dir, "stray_region0.ply")
    NPY_OUT = os.path.join(feat_out_dir, "RP_stray_region0.npy")

    # ====== 1) PLY 복사/저장 ======
    # open3d로 다시 저장해서 경로/포맷 안정화
    print("[1] writing PLY ->", PLY_OUT)

    # point cloud / mesh 둘 다 지원
    geom = o3d.io.read_point_cloud(PLY_IN)
    is_pcd = True
    if len(geom.points) == 0:
        # pointcloud로 안 읽히면 mesh로 시도
        is_pcd = False
        geom = o3d.io.read_triangle_mesh(PLY_IN)
        assert len(geom.vertices) > 0, "PLY seems empty (neither pointcloud nor mesh)"

    if is_pcd:
        N = np.asarray(geom.points).shape[0]
        o3d.io.write_point_cloud(PLY_OUT, geom, write_ascii=False, compressed=False)
        print("  - type: pointcloud, N(points) =", N)
    else:
        N = np.asarray(geom.vertices).shape[0]
        o3d.io.write_triangle_mesh(PLY_OUT, geom, write_ascii=False, compressed=False)
        print("  - type: mesh, N(vertices) =", N)

    # ====== 2) PT(압축) -> N x 512 로 풀어서 npy 저장 ======
    print("[2] converting features ->", NPY_OUT)
    d = torch.load(PT_IN, map_location="cpu")

    if "mask_full" not in d or "feat" not in d:
        raise KeyError(f"PT keys are {list(d.keys())}, expected at least ['mask_full','feat']")

    mask_full = d["mask_full"].cpu().numpy().astype(bool)  # (N,)
    feat_comp = d["feat"]  # (mask_sum, 512) float16/float32
    feat_comp = feat_comp.detach().cpu().float().numpy()

    if mask_full.shape[0] != N:
        raise RuntimeError(
            f"Mismatch: ply N={N}, mask_full N={mask_full.shape[0]}\n"
            f"- PLY_IN={PLY_IN}\n- PT_IN={PT_IN}\n"
            "같은 point ordering으로 만든 pt/ply가 맞는지 확인 필요"
        )

    D = feat_comp.shape[1]
    full_feat = np.zeros((N, D), dtype=np.float32)
    full_feat[mask_full] = feat_comp

    # (권장) feature normalize: 데모는 dot product 기반이라 정규화가 안정적
    # 관측된 점들만 정규화 (0벡터는 그대로 둠)
    eps = 1e-6
    norm = np.linalg.norm(full_feat, axis=1, keepdims=True)
    valid = norm[:, 0] > eps
    full_feat[valid] = full_feat[valid] / norm[valid]

    np.save(NPY_OUT, full_feat.astype(np.float32))
    print("  - saved:", NPY_OUT, "shape:", full_feat.shape, "dtype:", full_feat.dtype)
    print("  - observed ratio:", float(mask_full.mean()), "(", int(mask_full.sum()), "/", N, ")")

    print("\nDONE. Now run demo:")
    print("  cd", demo_root)
    print("  ./run_demo")

if __name__ == "__main__":
    main()
