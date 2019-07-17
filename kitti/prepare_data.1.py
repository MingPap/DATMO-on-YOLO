''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
import pickle as pickle
from kitti_object import *
import argparse


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds
     
# def demo():
#     import mayavi.mlab as mlab
#     from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
#     dataset = kitti_object(os.path.join('E:/MyFiles/desktop/data/object/data_object_image_2'))
#     data_idx = 0

#     # Load data from dataset
#     objects = dataset.get_label_objects(data_idx)
#     objects[0].print_object()
#     img = dataset.get_image(data_idx)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
#     img_height, img_width, img_channel = img.shape
#     print(('Image shape: ', img.shape))
#     pc_velo = dataset.get_lidar(data_idx)[:,0:3]
#     calib = dataset.get_calibration(data_idx)

#     ## Draw lidar in rect camera coord
#     #print(' -------- LiDAR points in rect camera coordination --------')
#     #pc_rect = calib.project_velo_to_rect(pc_velo)
#     #fig = draw_lidar_simple(pc_rect)
#     #raw_input()

#     # Draw 2d and 3d boxes on image
#     print(' -------- 2D/3D bounding boxes in images --------')
#     show_image_with_boxes(img, objects, calib)
#     raw_input()

#     # Show all LiDAR points. Draw 3d box in LiDAR point cloud
#     print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
#     #show_lidar_with_boxes(pc_velo, objects, calib)
#     #raw_input()
#     show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
#     raw_input()

#     # Visualize LiDAR points on images
#     print(' -------- LiDAR points projected to image plane --------')
#     show_lidar_on_image(pc_velo, img, calib, img_width, img_height) 
#     raw_input()
    
#     # Show LiDAR points that are in the 3d box
#     print(' -------- LiDAR points in a 3D bounding box --------')
#     box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P) 
#     box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
#     box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
#     print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

#     fig = mlab.figure(figure=None, bgcolor=(0,0,0),
#         fgcolor=None, engine=None, size=(1000, 500))
#     draw_lidar(box3droi_pc_velo, fig=fig)
#     draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
#     mlab.show(1)
#     raw_input()
    
#     # UVDepth Image and its backprojection to point clouds
#     print(' -------- LiDAR points in a frustum from a 2D box --------')
#     imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
#         calib, 0, 0, img_width, img_height, True)
#     imgfov_pts_2d = pts_2d[fov_inds,:]
#     imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

#     cameraUVDepth = np.zeros_like(imgfov_pc_rect)
#     cameraUVDepth[:,0:2] = imgfov_pts_2d
#     cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

#     # Show that the points are exactly the same
#     backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
#     print(imgfov_pc_velo[0:20])
#     print(backprojected_pc_velo[0:20])

#     fig = mlab.figure(figure=None, bgcolor=(0,0,0),
#         fgcolor=None, engine=None, size=(1000, 500))
#     draw_lidar(backprojected_pc_velo, fig=fig)
#     raw_input()

#     # Only display those points that fall into 2d box
#     print(' -------- LiDAR points in a frustum from a 2D box --------')
#     xmin,ymin,xmax,ymax = \
#         objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
#     boxfov_pc_velo = \
#         get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
#     print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

#     fig = mlab.figure(figure=None, bgcolor=(0,0,0),
#         fgcolor=None, engine=None, size=(1000, 500))
#     draw_lidar(boxfov_pc_velo, fig=fig)
#     mlab.show(1)
#     raw_input()



if __name__=='__main__':
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join('E:/MyFiles/desktop/data/object/data_object_image_2'))
    data_idx = 3

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    # # Draw lidar in rect camera coord
    # print(' -------- LiDAR points in rect camera coordination --------')
    # pc_rect = calib.project_velo_to_rect(pc_velo)
    # fig = draw_lidar_simple(pc_rect)
    # raw_input()

    # # Draw 2d and 3d boxes on image
    # print(' -------- 2D/3D bounding boxes in images --------')
    # show_image_with_boxes(img, objects, calib)
    # raw_input()


    # # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    # print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    # #show_lidar_with_boxes(pc_velo, objects, calib)
    # #raw_input()
    # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    # raw_input()

    # Visualize LiDAR points on images
    # print(' -------- LiDAR points projected to image plane --------')
    # img2 = show_lidar_on_image(pc_velo, img, calib, img_width, img_height) 
    # cv2.imwrite('messigray.png',img2)
    # raw_input() 

        # Visualize LiDAR points on images plane
    print(' -------- LiDAR points projected to image plane2 --------')
    img1 = np.zeros((img_height,img_width,img_channel), np.uint8)
    img2 = show_lidar_on_image(pc_velo, img1, calib, img_width, img_height) 
    cv2.imwrite('messigray.png',img2)
    raw_input() 
    
    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()
    
    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin,ymin,xmax,ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()


