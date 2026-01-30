import os
import numpy as np
import open3d as o3d

def main():
    # 입력들
    pth = "/home/yongbin53/yw/openscene/data/stray_processed/stray_3d/val/stray_scene.pth"
    pred_npy = "/home/yongbin53/yw/openscene/out/stray_lseg_fusion_demo/stray_scene_fusion_pred_idx.npy"

    # 출력
    out_ply = "/home/yongbin53/yw/openscene/out/stray_lseg_fusion_demo/stray_scene_fusion_floorfix.ply"
    out_npy = "/home/yongbin53/yw/openscene/out/stray_lseg_fusion_demo/stray_scene_fusion_floorfix_pred_idx.npy"

    assert os.path.exists(pth), pth
    assert os.path.exists(pred_npy), pred_npy

    xyz, rgb, _ = __import__("torch").load(pth)  # (N,3), rgb [-1,1] or (N,3)
    xyz = np.asarray(xyz, dtype=np.float64)

    pred = np.load(pred_npy).astype(np.int32)
    N = xyz.shape[0]
    assert pred.shape[0] == N, (pred.shape, N)

    # 1) 포인트클라우드 + 법선 추정
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 노멀: 울퉁불퉁 바닥이면 이웃 반경을 너무 작게 잡으면 노이즈 커짐
    # voxel_size=0.02 기준이면 radius 0.08~0.12 정도가 안정적
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30))
    normals = np.asarray(pcd.normals)

    # 2) "up" 방향 가정
    # Stray world 좌표의 up이 정확히 z라고 단정 못해도,
    # 보통 스캔 데이터는 z-up인 경우가 많고, 아니면 PCA로 보정 가능.
    # 여기서는 z-up 가정 + 낮은 높이 기반으로 robust하게 잡는다.
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # 노멀 방향은 +/-가 섞일 수 있으니 abs(dot) 사용
    cos = np.abs(normals @ up)  # 1이면 수평면(노멀이 up에 평행)
    # 수평면 후보: 15~25도 이내 정도
    cos_thr = np.cos(np.deg2rad(25.0))
    horizontal = cos >= cos_thr

    # 3) 낮은 높이 후보(바닥은 보통 가장 낮은 쪽)
    z = xyz[:, 2]
    z0 = np.percentile(z, 15)  # 바닥이 울퉁불퉁하면 5%는 너무 빡셀 수 있어 15%부터
    low = z <= z0

    candidate = horizontal & low

    # 4) 후보 중에서 "가장 큰 연결 성분"만 바닥으로 선택 (울퉁불퉁해도 연결되면 잡힘)
    # neighbor graph 만들기
    # Open3D의 클러스터링을 쓰려면 voxel downsample 후 연결 성분을 구하는 게 더 안정적이지만
    # 간단하게 radius neighbor 기반으로 근사한다.
    idx = np.where(candidate)[0]
    if len(idx) < 1000:
        print("candidate too small:", len(idx), "-> relax thresholds")
        # 완화
        z0 = np.percentile(z, 25)
        low = z <= z0
        cos_thr = np.cos(np.deg2rad(35.0))
        horizontal = cos >= cos_thr
        candidate = horizontal & low
        idx = np.where(candidate)[0]

    cand_pcd = o3d.geometry.PointCloud()
    cand_pcd.points = o3d.utility.Vector3dVector(xyz[idx])
    # DBSCAN으로 큰 덩어리 찾기(바닥이 조각나 있어도 “큰 덩어리”가 바닥일 가능성 큼)
    labels = np.array(cand_pcd.cluster_dbscan(eps=0.08, min_points=50, print_progress=False))
    if labels.max() < 0:
        print("DBSCAN failed -> use all candidates")
        floor_idx = idx
    else:
        # 가장 큰 클러스터 선택
        best = np.bincount(labels[labels >= 0]).argmax()
        floor_local = np.where(labels == best)[0]
        floor_idx = idx[floor_local]

    # 5) floor 주변을 조금 확장(울퉁불퉁/틈 메우기)
    # 바닥 점들로부터 kNN / radius로 확장
    tree = o3d.geometry.KDTreeFlann(pcd)
    expand = np.zeros(N, dtype=bool)
    for pid in floor_idx[::20]:  # 전부 돌리면 느릴 수 있으니 subsample
        _, nn, _ = tree.search_radius_vector_3d(pcd.points[pid], 0.06)
        expand[nn] = True

    # 확장 조건: 높이가 너무 높으면 제외(계단/테이블 상판 방지)
    z_floor_med = np.median(z[floor_idx])
    expand = expand & (z <= z_floor_med + 0.25)  # 25cm 위까지는 바닥 굴곡으로 허용

    floor_mask = np.zeros(N, dtype=bool)
    floor_mask[floor_idx] = True
    floor_mask = floor_mask | expand

    # 6) pred에서 floor 인덱스가 무엇인지 알아야 함
    # matterport labelset에서 floor가 몇 번인지는 설정에 따라 달라질 수 있으니,
    # 일단 “현재 pred에서 floor로 쓰이는 색/인덱스”를 너가 확인해줘야 완벽함.
    # 여기서는 보통 matterport21에서 floor가 존재한다고 가정하고,
    # floor_idx_guess를 1차로 “현재 pred에서 바닥 후보에서 가장 많이 나온 클래스”로 잡는다.
    floor_idx_guess = np.bincount(pred[floor_mask]).argmax()
    print("auto floor class idx (from candidates):", floor_idx_guess)

    pred2 = pred.copy()
    pred2[floor_mask] = floor_idx_guess

    np.save(out_npy, pred2)
    print("saved:", out_npy)

    # 7) 색 입혀서 ply 저장(간단: floor만 초록, 나머지 원본 rgb)
    # 원본 rgb가 [-1,1]이면 [0,1]로 변환
    if rgb is None or (isinstance(rgb, np.ndarray) and rgb.shape[1] != 3):
        colors = np.zeros((N,3), dtype=np.float64)
    else:
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.min() < 0:
            colors = (rgb + 1.0) / 2.0
        else:
            colors = np.clip(rgb, 0.0, 1.0)

    colors[floor_mask] = np.array([0.0, 1.0, 0.0])  # floor 후보만 초록
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(xyz)
    out_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_ply, out_pcd)
    print("saved:", out_ply)

if __name__ == "__main__":
    main()
