import argparse
import numpy as np
import open3d as o3d

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_ply", required=True, help="input mesh ply (same mesh used for prior)")
    ap.add_argument("--prior_npy", required=True, help="prior npy (uint8 0/1) length == #mesh vertices")
    ap.add_argument("--out_ply", required=True, help="output ply with vertex colors")
    ap.add_argument("--base_gray", type=float, default=0.35, help="base gray for non-floor vertices [0~1]")
    ap.add_argument("--floor_green", type=float, nargs=3, default=[0.0, 1.0, 0.0], help="RGB for floor vertices")
    return ap.parse_args()

def main():
    args = parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh_ply)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to read mesh: {args.mesh_ply}")

    V = np.asarray(mesh.vertices)
    prior = np.load(args.prior_npy)

    # Flatten prior (supports (N,), (N,1), (1,N))
    prior = np.asarray(prior).reshape(-1)
    if prior.shape[0] != V.shape[0]:
        raise RuntimeError(
            f"prior length mismatch: prior={prior.shape[0]} vs mesh vertices={V.shape[0]}"
        )

    # Ensure vertex colors exist
    base = float(args.base_gray)
    colors = np.full((V.shape[0], 3), base, dtype=np.float64)

    floor_rgb = np.array(args.floor_green, dtype=np.float64).reshape(1, 3)
    floor_mask = (prior > 0)
    colors[floor_mask] = floor_rgb

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_triangle_mesh(args.out_ply, mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write: {args.out_ply}")

    floor_cnt = int(floor_mask.sum())
    print(f"Saved: {args.out_ply}")
    print(f"mesh vertices: {V.shape[0]}")
    print(f"floor prior positives: {floor_cnt} ({100.0*floor_cnt/V.shape[0]:.2f}%)")

if __name__ == "__main__":
    main()
