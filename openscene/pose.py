import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation
import os
vis = np.zeros((480, 640, 3), dtype=np.uint8)  # <- 이게 빠져 있었음

out = "/home/yongbin53/yw/openscene/tmp/pose_sanity.png"
os.makedirs(os.path.dirname(out), exist_ok=True)

ok = cv2.imwrite(out, vis)
print("imwrite ok?", ok, "->", out)

# paths
SCENE = "/home/yongbin53/yw/openscene/stray_data"
rgb_path = f"{SCENE}/rgb_frames/000100.png"
odom_path = f"{SCENE}/odometry.csv"
mesh_path = f"{SCENE}/mesh_refined.ply"
K = np.loadtxt(f"{SCENE}/camera_matrix.csv", delimiter=",")

# load rgb
img = cv2.imread(rgb_path)
H, W = img.shape[:2]

# load mesh
mesh = o3d.io.read_triangle_mesh(mesh_path)
V = np.asarray(mesh.vertices)

# load odometry (frame 100)
odom = np.loadtxt(odom_path, delimiter=",", skiprows=1)
line = odom[100]
pos = line[2:5]
quat = line[5:]

T_WC = np.eye(4)
T_WC[:3,:3] = Rotation.from_quat(quat).as_matrix()
T_WC[:3,3] = pos

T_CW = np.linalg.inv(T_WC)

# sample some vertices
idx = np.random.choice(len(V), 3000, replace=False)
pts_world = V[idx]

# world -> camera
pts_h = np.hstack([pts_world, np.ones((len(pts_world),1))])
pts_cam = (T_CW @ pts_h.T).T[:, :3]

# keep points in front of camera
mask = pts_cam[:,2] > 0.1
pts_cam = pts_cam[mask]

# project
u = (K[0,0] * pts_cam[:,0] / pts_cam[:,2] + K[0,2]).astype(int)
v = (K[1,1] * pts_cam[:,1] / pts_cam[:,2] + K[1,2]).astype(int)

valid = (u>=0)&(u<W)&(v>=0)&(v<H)
u, v = u[valid], v[valid]

# draw
vis = img.copy()
for x,y in zip(u,v):
    cv2.circle(vis, (x,y), 1, (0,255,0), -1)

out = "/home/yongbin53/yw/openscene/tmp/pose_sanity.png"
cv2.imwrite(out, vis)
print("saved:", out)
