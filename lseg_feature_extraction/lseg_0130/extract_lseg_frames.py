import os, glob
import numpy as np
import torch
import torchvision.transforms as T

from lseg_feature_extraction import LSegModule
from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from fusion_util import extract_lseg_img_feature

def build_evaluator(ckpt_path: str, img_long_side=256, device="cuda"):
    IMG_LONG_SIDE = img_long_side
    CROP_BASE = 2 * IMG_LONG_SIDE  # 512

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        inference_only=True,
        data_path="../datasets/",
        dataset="ade20k",
        backbone="clip_vitl16_384",
        aux=False, num_features=256, aux_weight=0,
        se_loss=False, se_weight=0,
        base_lr=0, batch_size=1, max_epochs=0,
        ignore_index=255, dropout=0.0,
        scale_inv=False, augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=True,
        map_locatin="cpu",
        arch_option=0, block_depth=0, activation="lrelu",
    )

    model = module.net if isinstance(module.net, BaseNet) else module
    model = model.eval().to(device)

    # important: set crop/base to match this dataset
    model.crop_size = CROP_BASE
    model.base_size = CROP_BASE

    model.mean=[0.5,0.5,0.5]; model.std=[0.5,0.5,0.5]

    evaluator = LSeg_MultiEvalModule(model, scales=[1], flip=False).to(device)
    evaluator.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    return evaluator, transform

def main():
    SCENE="/home/yongbin53/yw/openscene/data/stray_data"
    CKPT="/home/yongbin53/yw/lseg_feature_extraction/checkpoints/demo_e200.ckpt"
    OUT_DIR="/home/yongbin53/yw/openscene/stray_data_lseg/lseg_frames_512"  # cache
    os.makedirs(OUT_DIR, exist_ok=True)

    rgb_dir=os.path.join(SCENE, "rgb_frames")
    rgb_list=sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    assert len(rgb_list) > 0, "no rgb frames"

    device="cuda" if torch.cuda.is_available() else "cpu"
    evaluator, transform = build_evaluator(CKPT, img_long_side=256, device=device)

    # 원하는 만큼만 먼저 뽑아보기 (안전)
    # N = min(50, len(rgb_list))  # 먼저 50장
    N = len(rgb_list)
    print("total:", len(rgb_list), "extract:", N, "device:", device)

    for i in range(N):
        img_path = rgb_list[i]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUT_DIR, f"{stem}.npy")
        if os.path.exists(out_path):
            continue

        feat = extract_lseg_img_feature(img_path, transform, evaluator, label="")  # (C,H,W) fp16 cuda
        feat_np = feat.detach().float().cpu().numpy()  # 저장은 float32 추천(호환 안정)
        np.save(out_path, feat_np)
        if i % 10 == 0:
            print("saved", out_path, feat_np.shape, feat_np.dtype)

    print("done ->", OUT_DIR)

if __name__ == "__main__":
    main()
