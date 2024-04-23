# from asyncore import poll3
from cmath import pi
from psutil import cpu_percent
import numpy as np
from numpy.linalg import norm
# from ar_robot_skills.utils.to_quaternion import to_3Dvector_quaternion
# from ar_toolkit.utils.math.conversions.vector3D_axangle_to_matrix import vector3D_axangle_to_matrix
# from ar_toolkit.utils.math.conversions.vector3D_quaternion_to_matrix import vector3D_quaternion_to_matrix
# from ar_toolkit.utils.math.conversions.vector3D_axangle_to_vector3D_quaternion import vector3D_axangle_to_vector3D_quaternion
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import math
import transforms3d

import matplotlib as mpl
# import ar_toolkit.robots as robots


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def define_input( p_start, p_mid, p_end):
    """
    Returns the center of the circle, rotation axis and span angle with the given 3 points.
    """
    x0, y0, z0 = p_start
    x1, y1, z1 = p_mid
    x2, y2, z2 = p_end

    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

    u_cross_v= [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx] # z axis
    e3 = u_cross_v/norm(u_cross_v)

    ###### circle center
    ## 1. rotation matrix
    P12 = [x1-x0, y1-y0, z1-z0]
    e1 = P12/norm( P12 )
    e2 = - np.cross(e1, e3)
    R = np.zeros((3,3))
    R[:,0] = e1
    R[:,1] = e2
    R[:,2] = e3
    R_inv = np.linalg.inv(R)
   
    P2 = np.dot(R_inv, P12 )
    P13 = [x2-x0, y2-y0, z2-z0]
    P3 = np.dot(R_inv, P13 )


    ## 2. circle center
    center, radius = define_circle((0,0), (P2[0],P2[1]), (P3[0],P3[1]))
    C = [center[0], center[1],  0]
    circle_center = np.dot(R, C) + [x0, y0, z0]

    ##### span angle
    v1 = [x0-circle_center[0], y0-circle_center[1], z0-circle_center[2]]
    v2 = [x1-circle_center[0], y1-circle_center[1], z1-circle_center[2]]
    v3 = [x2-circle_center[0], y2-circle_center[1], z2-circle_center[2]]
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    unit_v3 = v3 / np.linalg.norm(v3)
    dot_product12 = np.dot(unit_v1, unit_v2)
    angle12 = np.arccos(dot_product12)    
    dot_product23 = np.dot(unit_v2, unit_v3)
    angle23 = np.arccos(dot_product23)
    angle = angle12 + angle23
    print("angle", angle)

    return circle_center, e3, angle*180/pi

def circular_trajectory(sphere_center, start_point, spanAngle, steps, rotateAxis, init_quat):
    """Method to generate a 3d circular trajectory

        Args:
            sphere_center(array): center of the circular trajectory, normally stands for the object to be inspected
            spanAngle(float) is given in degree
            start_point(array): EEF's initial cartesian position
            steps(integar): the number of points in the trajectory
            rotateAxis(array)
            init_quat(array): EEF's initial pose

        Returns:
            list: generated set of points and quaternion

    """
    quat = [1.0, 0.0, 0.0, 0.0]
    quat = np.array(quat)
    start_point = np.array(start_point), init_quat

    ## angle check
    if spanAngle % 180 == 0:
        spanAngle = spanAngle - 1
    spanAngle = math.pi*spanAngle/180 # spanAngle in [Rad]

    
    ## calculate stop point
    # axangle to rotation matrix
    rotateAxis = rotateAxis/norm(rotateAxis)
    SP_s = start_point[0]-  sphere_center 
    rotation_matrix_for_stop_point = transforms3d.axangles.axangle2mat(rotateAxis, spanAngle)
    stop_point = np.dot(rotation_matrix_for_stop_point, SP_s) + sphere_center
    stop_point = np.array(stop_point), start_point[1]

    ##  get circle center
    v1 = (start_point[0] + stop_point[0])/2
    v2 = v1 - sphere_center
    l3 = np.dot(rotateAxis, v2)
    v4 = l3 * rotateAxis
    circle_center = sphere_center + v4
    
    ##  trajectory generation
    CP_s = start_point[0] - circle_center 
    CP_e = stop_point[0] - circle_center

    final_points = []
    final_quat = []
    final_traj = []
    
    for i in range(steps+1):
        theta = i*(spanAngle/steps)
        ## PART1 coordinates
        CP_i = CP_s * math.sin(spanAngle-theta)/math.sin(spanAngle) + CP_e * math.sin(theta)/math.sin(spanAngle)
        point = CP_i + circle_center
        final_points.append(point)

        ## PART2 quaternion
        rot_i_1 = transforms3d.quaternions.quat2mat(init_quat)
        quat_i = transforms3d.quaternions.axangle2quat(rotateAxis, theta)
        rot_i_2 = transforms3d.quaternions.quat2mat(quat_i)
        rot_i = np.dot(rot_i_2, rot_i_1)
        quat_i = transforms3d.quaternions.mat2quat(rot_i)
        final_quat.append(quat_i)

        ## PART3 trajectory
        final_traj.append((point, quat_i))
        print("final_traj[", i , "] = ", final_traj[i])
   
    return final_traj 
