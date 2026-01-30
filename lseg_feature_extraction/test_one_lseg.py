# ~/yw/lseg_feature_extraction/test_one_lseg.py
import os
import torch
import torchvision.transforms as T

from lseg_feature_extraction import LSegModule
from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from fusion_util import extract_lseg_img_feature

def main():
    img_path="/home/yongbin53/yw/openscene/stray_data/rgb_frames/000000.png"
    ckpt="/home/yongbin53/yw/lseg_feature_extraction/checkpoints/demo_e200.ckpt"
    assert os.path.exists(img_path), img_path
    assert os.path.exists(ckpt), ckpt

    # 이미지 long side (256x192 -> 256)
    IMG_LONG_SIDE = 256
    CROP_BASE = 2 * IMG_LONG_SIDE   # repo 의도대로

    # 1) ckpt load (inference only)
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=ckpt,
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
        widehead_hr=True,  # ★ 중요: 출력 해상도 맞추기 시도
        map_locatin="cpu",
        arch_option=0, block_depth=0, activation="lrelu",
    )

    model = module.net if isinstance(module.net, BaseNet) else module
    model = model.eval().cuda()

    # ★ 중요: crop/base 강제 세팅
    model.crop_size = CROP_BASE
    model.base_size = CROP_BASE

    model.mean=[0.5,0.5,0.5]; model.std=[0.5,0.5,0.5]

    evaluator = LSeg_MultiEvalModule(model, scales=[1], flip=False).cuda()
    evaluator.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])

    feat = extract_lseg_img_feature(img_path, transform, evaluator, label="")

    print("feat type:", type(feat))
    if torch.is_tensor(feat):
        print("feat.shape:", tuple(feat.shape), "dtype:", feat.dtype, "device:", feat.device)
        # 값 sanity
        print("min/max:", float(feat.min()), float(feat.max()))
    else:
        import numpy as np
        arr = feat if isinstance(feat, np.ndarray) else feat.detach().cpu().numpy()
        print("feat.shape:", arr.shape, "dtype:", arr.dtype)
        print("min/max:", float(arr.min()), float(arr.max()))

if __name__ == "__main__":
    main()
