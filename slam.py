import Class_Code.Triangulate as triangulate
import numpy as np
import open3d as o3d

Rot = []

def run_slam(image, PCL, Cameras, Image_Library):
    Image_Library.append(image)
    if len(Image_Library) == 1:
        return None, Cameras, Image_Library

    image1 = Image_Library[-1]
    image2 = Image_Library[-2]
    triangle_points, E = triangulate.triangulate(image1, image2)

    # Find cameras
    U, d, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T = np.matmul(np.matmul(U, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])), np.transpose(U))
    R = np.matmul(np.matmul(U, np.transpose(W)), Vt)
    t = np.array([[T[2][1]], [T[0][2]], [T[1][0]]])
    Rot.append(R)
    H = np.append(R, t, axis=1)
    H = np.vstack([H, [0, 0, 0, 1]])

    C_next = np.matmul(H, np.transpose(np.append(Cameras[-1], [1])))
    scale_adjustment = np.array([-C_next[0], -C_next[1], C_next[2] * 0.75])
    Cameras = np.vstack([Cameras, scale_adjustment])

    if PCL is None:
        PCL = o3d.geometry.PointCloud()
        PCL.points = o3d.utility.Vector3dVector(np.transpose(triangle_points[0:3, :]))
        return PCL, Cameras, Image_Library
    else:
        threshold = 0.02
        trans_init = np.asarray([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.transpose(triangle_points[0:3,:]))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd, PCL, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        pcd.transform(reg_p2p.transformation)
        return pcd, Cameras, Image_Library

