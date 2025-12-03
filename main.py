import numpy as np
import open3d as o3d
import pickle as pkl
import trimesh
import cv2
import argparse
import os

def load_mesh(fn):
    obj = trimesh.load(fn)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(obj.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(obj.faces)
    mesh.compute_vertex_normals()
    return mesh

def load_params(fn):
    with open(fn, 'rb') as f:
        params = pkl.load(f)
    return params

def projection(point, K):
    # camera extrinsics: R=I, T=0
    point_cam = point.reshape(3, 1)
    point_2d_homogeneous = K @ point_cam  # (3, 1)
    point_2d = point_2d_homogeneous[:2, 0] / point_2d_homogeneous[2, 0]
    return point_2d

def point_index_set(picked_points):
    """Extracts a set of vertex indices from picked points."""
    index_set = set()
    for p in picked_points:
        index_set.add(p.index)
    return index_set

def run(dir):
    mesh = load_mesh(os.path.join(dir, '000.obj'))
    mesh.paint_uniform_color([0.7, 0.0, 0.0])

    params = load_params(os.path.join(dir, '000.pkl'))
    R = params['camera_rotation'].reshape(3,3)
    T = params['camera_translation'].reshape(3)
    H, W = params['size']
    focal_length = params['focal_length']
    K = np.array([[focal_length, 0, W/2],
                  [0, focal_length, H/2],
                  [0, 0, 1]])

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name='Vertex Annotation Tool', width=W, height=H)

    mesh.rotate(np.diag([1, -1, -1.] @ R), center=(0, 0, 0))
    mesh.translate(np.diag([1, 1, 1.]) @ T)
    
    vis.add_geometry(mesh, reset_bounding_box=True)

    view_ctrl = vis.get_view_control()
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H,
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2]
    )
    camera_params.extrinsic = np.eye(4) 
    view_ctrl.convert_from_pinhole_camera_parameters(camera_params, True)

    img = cv2.imread(os.path.join(dir, '000.jpg'))
    img_copy = img.copy() 
    cv2.imshow('Image', img_copy)

    last_point_index_set = set()

    while True:
        vis.poll_events()
        vis.update_renderer()
        
        current_point_index_set = point_index_set(vis.get_picked_points())
        
        if current_point_index_set != last_point_index_set:
            img_copy = img.copy()
            for idx in current_point_index_set:
                point_3d = np.asarray(mesh.vertices)[idx]
                point_2d = projection(np.array(point_3d), K) 
                x, y = int(point_2d[0]), int(point_2d[1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
                
            last_point_index_set = current_point_index_set
            cv2.imshow('Image', img_copy)
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    vis.destroy_window()
    cv2.destroyAllWindows()

    with open(os.path.join(dir, 'annotated_vertices.txt'), 'w') as f:
        for idx in last_point_index_set:
            line = f"{int(idx)}"
            f.write(line + '\n')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    run(args.dir)