# /home/yongbin53/yw/openscene/tools/make_stray_pth.py
import os
import torch
import numpy as np
import open3d as o3d

def main():
    ply = "/home/yongbin53/yw/openscene/data/stray_data/pointcloud_refined.ply"
    out_root = "/home/yongbin53/yw/openscene/data/stray_processed/stray_3d/val"
    scene_name = "stray_scene"
    os.makedirs(out_root, exist_ok=True)

    pcd = o3d.io.read_point_cloud(ply)
    xyz = np.asarray(pcd.points).astype(np.float32)

    if pcd.has_colors():
        rgb = np.asarray(pcd.colors).astype(np.float32)
        rgb = rgb * 2.0 - 1.0  # [-1, 1]
    else:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    # GT 없으니 255 (unlabeled)
    label = np.full((xyz.shape[0],), 255, dtype=np.uint8)  # ★ uint8로 고정

    out_path = os.path.join(out_root, f"{scene_name}.pth")

    # ★ 핵심: torch tensor로 감싸지 말고 "numpy 그대로" 저장
    torch.save((xyz, rgb, label), out_path)

    print("saved:", out_path)
    print("xyz:", xyz.shape, xyz.dtype, "rgb:", rgb.shape, rgb.dtype, "label:", label.shape, label.dtype)

if __name__ == "__main__":
    main()
