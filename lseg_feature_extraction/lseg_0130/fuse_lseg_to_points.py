import os, glob
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio

from fusion_util import make_intrinsic, PointCloudToImageMapper

def load_odometry_csv(path):
    # columns: timestamp, frame, x,y,z,qx,qy,qz,qw
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # ensure sorted by frame index
    data = data[np.argsort(data[:,1])]
    return data

def T_WC_from_row(row):
    pos = row[2:5]
    quat = row[5:9]
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = Rotation.from_quat(quat).as_matrix()
    T[:3,3] = pos
    return T

def main():
    SCENE="/home/yongbin53/yw/openscene/data/stray_data"
    PC_PATH=os.path.join(SCENE, "pointcloud_refined.ply")
    ODOM_PATH=os.path.join(SCENE, "odometry.csv")
    K_PATH=os.path.join(SCENE, "camera_matrix.csv")
    DEPTH_DIR=os.path.join(SCENE, "depth")
    FEAT_DIR="/home/yongbin53/yw/openscene/stray_data_lseg/lseg_frames_512"   # extract_lseg_frames.py output

    OUT_DIR="/home/yongbin53/yw/openscene/stray_data_lseg/lseg_fused_feature_512"
    os.makedirs(OUT_DIR, exist_ok=True)

    # load point cloud
    pc = o3d.io.read_point_cloud(PC_PATH)
    pts = np.asarray(pc.points).astype(np.float64)
    N = pts.shape[0]
    print("points:", N)

    # load intrinsics
    K = np.loadtxt(K_PATH, delimiter=",").astype(np.float64)
    # mapper wants 4x4 intrinsic in the repo’s convention
    intrinsic_4 = make_intrinsic(K[0,0], K[1,1], K[0,2], K[1,2])

    # image size (W,H)
    W, H = 256, 192
    mapper = PointCloudToImageMapper(image_dim=(W, H), visibility_threshold=0.25, cut_bound=0, intrinsics=intrinsic_4)

    # feature bank
    C = 512
    feat_sum = np.zeros((N, C), dtype=np.float32)
    feat_cnt = np.zeros((N,), dtype=np.int32)

    odom = load_odometry_csv(ODOM_PATH)
    print("odom rows:", len(odom))

    # 먼저 일부 프레임만으로 테스트 (속도/정합 확인)
    MAX_FRAMES = len(odom)
    step = 5  # 0,5,10,... 이렇게만
    used = 0

    for idx in range(0, min(len(odom), MAX_FRAMES), step):
        row = odom[idx]
        frame = int(row[1])

        feat_path = os.path.join(FEAT_DIR, f"{frame:06d}.npy")
        depth_path = os.path.join(DEPTH_DIR, f"{frame:06d}.png")

        if (not os.path.exists(feat_path)) or (not os.path.exists(depth_path)):
            continue

        # load 2D feat (C,H,W)
        feat2d = np.load(feat_path)  # float32
        assert feat2d.shape[0] == C, feat2d.shape
        assert feat2d.shape[1] == H and feat2d.shape[2] == W, feat2d.shape

        # load depth (H,W), unit? your earlier min/max ~ 2610 -> likely millimeters
        depth = imageio.imread(depth_path).astype(np.float32)
        if depth.max() > 100:  # heuristic: mm -> meters
            depth = depth / 1000.0

        # pose T_WC (camera->world)
        T_WC = T_WC_from_row(row)

        # compute mapping (for each point -> (v,u,mask))
        mapping = mapper.compute_mapping(T_WC, pts, depth=depth, intrinsic=intrinsic_4)  # (N,3): (v,u,mask)
        m = mapping[:,2].astype(bool)
        if m.sum() == 0:
            continue

        vv = mapping[m,0].astype(np.int64)
        uu = mapping[m,1].astype(np.int64)

        # gather pixel features -> (num_visible, C)
        pix_feat = feat2d[:, vv, uu].T  # (M,C)

        # accumulate
        feat_sum[m] += pix_feat
        feat_cnt[m] += 1

        used += 1
        print(f"[{used}] frame={frame} visible={m.sum()}")

    # finalize mean
    valid = feat_cnt > 0
    feat_mean = np.zeros_like(feat_sum, dtype=np.float32)
    feat_mean[valid] = feat_sum[valid] / feat_cnt[valid, None]

    # save as torch .pt (OpenScene style와 맞추기 쉬움)
    import torch
    out_pt = os.path.join(OUT_DIR, "pointcloud_feat_lseg512.pt")
    torch.save({
        "feat": torch.from_numpy(feat_mean),      # (N,512) float32
        "count": torch.from_numpy(feat_cnt),      # (N,)
        "valid": torch.from_numpy(valid),         # (N,)
    }, out_pt)
    print("saved:", out_pt)

if __name__ == "__main__":
    main()
