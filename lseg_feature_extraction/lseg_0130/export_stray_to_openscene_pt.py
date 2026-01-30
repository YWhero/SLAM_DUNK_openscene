import os
import torch
from types import SimpleNamespace
from fusion_util import save_fused_feature

def main():
    # 입력: 네가 이미 만든 fused 결과
    inp = "/home/yongbin53/yw/openscene/stray_data_lseg/lseg_fused_feature_512/pointcloud_feat_lseg512.pt"
    d = torch.load(inp, map_location="cpu")

    # d 안에 뭐가 있는지 네 구현에 따라 다를 수 있는데,
    # 너 로그로는 최소 count가 있고, 보통 feat_sum 또는 feat(평균)가 있을 것.
    # 아래 키는 상황에 맞춰 한 번만 맞추면 됨.
    if "feat" in d:
        feat_bank = d["feat"]          # (N, 512) 또는 (N,512) 평균 feature
    elif "feat_bank" in d:
        feat_bank = d["feat_bank"]
    elif "sum" in d and "count" in d:
        feat_bank = d["sum"] / d["count"].clamp(min=1)  # (N,512)
    else:
        raise KeyError(f"cannot find feature tensor keys in {list(d.keys())}")

    count = d["count"].view(-1)  # (N,)
    n_points = count.numel()

    # point_ids: 관측된 포인트들
    point_ids = torch.nonzero(count > 0, as_tuple=False).view(-1)

    # 출력 폴더 (OpenScene에서 읽기 쉬운 위치로)
    out_dir = "/home/yongbin53/yw/openscene/stray_data_lseg/openscene_input_lseg512"
    os.makedirs(out_dir, exist_ok=True)

    # save_fused_feature가 args를 요구하므로 최소 args만 만들어줌
    args = SimpleNamespace()
    args.num_rand_file_per_scene = 1
    args.n_split_points = 2_000_000  # 너 포인트 60만이므로 충분히 큼

    scene_id = "stray_scene"  # 원하는 이름으로

    save_fused_feature(
        feat_bank=feat_bank,
        point_ids=point_ids,
        n_points=n_points,
        out_dir=out_dir,
        scene_id=scene_id,
        args=args
    )

    print("done ->", out_dir)

if __name__ == "__main__":
    main()
