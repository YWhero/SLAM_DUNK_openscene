import numpy as np

path = "/home/yongbin53/yw/openscene/stray_data_clipb32/clip_frames_512/000000.npy"

arr = np.load(path)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min / max:", arr.min(), arr.max())
