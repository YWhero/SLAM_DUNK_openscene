# /home/yongbin53/yw/openscene/demo/check_multi_query_sim.py
import os
import numpy as np
import torch

PT = "/home/yongbin53/yw/openscene/stray_data_clipb32/clip_fused_feature_512/pointcloud_feat_clipb32_512.pt"
TMP_DIR = "/home/yongbin53/yw/openscene/demo/tmp"

QUERIES = [
    "chair", "bed",
    "floor", "wall",
    "glass"
]

def l2norm_rows(X: np.ndarray, eps=1e-6):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

def main():
    assert os.path.exists(PT), PT
    d = torch.load(PT, map_location="cpu")
    F = d["feat"].float().numpy()          # (N,512)
    cnt = d["count"].numpy()
    valid = cnt > 0

    # point feature normalize (반드시)
    Fn = l2norm_rows(F)

    print("N:", F.shape[0], "valid_ratio:", float(valid.mean()))
    print("count mean(valid):", float(cnt[valid].mean()), "max:", int(cnt.max()))
    print()

    for q in QUERIES:
        tpath = os.path.join(TMP_DIR, f"{q}.npy")
        if not os.path.exists(tpath):
            print(f"[skip] tmp not found for query='{q}' -> {tpath}")
            print("  (demo에서 한번 쿼리를 입력하거나, clip_server로 쿼리를 보내서 npy를 먼저 생성해야 함)")
            print()
            continue

        T = np.load(tpath).astype("float32")[0]  # (512,)
        T = T / (np.linalg.norm(T) + 1e-6)

        sim = Fn[valid] @ T

        mn, mean, mx = float(sim.min()), float(sim.mean()), float(sim.max())
        print(f"== {q} ==")
        print("  sim min/mean/max:", mn, mean, mx)

        # percentiles (색칠 기준 잡을 때 핵심)
        for p in [50, 75, 90, 95, 97, 99, 99.5, 99.9]:
            print(f"  p{p:>4}: {float(np.percentile(sim, p)):.6f}")
        print()

if __name__ == "__main__":
    main()
