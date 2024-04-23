
from psutil import cpu_percent
import numpy as np
from numpy.linalg import norm
from utils.math.conversions.vector3D_quaternion_to_matrix import vector3D_quaternion_to_matrix
import matplotlib.pyplot as plot
import math
import transforms3d

import matplotlib as mpl



def circular_trajectory(sphere_center, start_point, spanAngle, steps, rotateAxis):
    """Method to generate a 3d circular trajectory

        Args:
            sphere_center(array): the center of sphere 
            start_point(array): start point of end effector
            spanAngle(float) is given in degree
            steps: number of points generated in trajectory is steps+1 including start point
            rotateAxis(array): default rotation axis is z-axis

        Returns:
            list: generated set of points and quaternion

    """

    ## input check
    # if len(start_point) != 1:
    #     raise ValueError("The number of point given should equal to 1")
    
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
    # print("start_point:", start_point)
    # print("stop_point:", stop_point)
    # print("circle_center:", circle_center)
    
    ##  trajectory generation
    CP_s = start_point[0]-  circle_center 
    CP_e = stop_point[0] -  circle_center

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
        quat_i = transforms3d.quaternions.axangle2quat(rotateAxis, theta)
        final_quat.append(quat_i)
        
        ## PART3 trajectory
        final_traj.append((point, quat_i))
        # print("final_traj[", i , "] = ", final_traj[i])
   
    return final_traj, circle_center, rotateAxis



# function to draw the trajectory, with: points as list
def draw_trajectory(final_traj, circle_center, sphere_center,start_point, rotateAxis):
    
    mpl.rcParams['legend.fontsize'] = 10
    fig = plot.figure()
    ax = fig.add_subplot(projection='3d')
    limt = 2
    ax.set_xlim([-limt, limt])
    ax.set_ylim([-limt, limt])
    ax.set_zlim([-limt, limt])
    ax.set_aspect('equal', adjustable='box')

    x = []
    y = []
    z = []

    for i in range(0, len(final_traj)):
        x.append(final_traj[i][0][0])
        y.append(final_traj[i][0][1])
        z.append(final_traj[i][0][2])
        if i % 5 == 0:
            draw_coordinate_system(final_traj[i], ax)
        # print("final_traj[", i , "] = ", final_traj[i])

    ax.plot(x, y, z, label='trajectory',  color="red")

    #######################################################
    ##############    SOME NICE DRAWING    ################
    #######################################################
    x0 = []
    y0 = []
    z0 = []
    x0.append(sphere_center[0])
    y0.append(sphere_center[1])
    z0.append(sphere_center[2])
    x0.append(start_point[0][0])
    y0.append(start_point[0][1])
    z0.append(start_point[0][2])    
    stop_point = final_traj[len(final_traj)-1]
    x0.append(stop_point[0][0])
    y0.append(stop_point[0][1])
    z0.append(stop_point[0][2])
    x0.append(circle_center[0])
    y0.append(circle_center[1])
    z0.append(circle_center[2])
    ax.plot(x0,y0,z0, marker='o', markersize=3,  label='crucial poitns', color="black")
  
  
    ax.quiver(sphere_center[0], sphere_center[1], sphere_center[2],\
        rotateAxis[0], rotateAxis[1], rotateAxis[2], arrow_length_ratio=0.3,label='Rotation Axis', color="green")

    #######################################################
    #######################################################

    ax.legend()
    plot.show()


def draw_coordinate_system(coordinate, ax):
    x_i = coordinate[0][0]
    y_i = coordinate[0][1]
    z_i = coordinate[0][2]

    rotation_matrix = transforms3d.quaternions.quat2mat(coordinate[1])
    dx_i = rotation_matrix[:,0]
    dy_i = rotation_matrix[:,1]
    dz_i = rotation_matrix[:,2]
    # print("rotation_matrix:", rotation_matrix)
    # print("dx_i:", dx_i)
    # print("dy_i:", dy_i)
    # print("dz_i:", dz_i)

    ax.quiver(x_i, y_i, z_i, dx_i[0], dx_i[1], dx_i[2],
    arrow_length_ratio=0.1, color="green") # x axis is marked as green
    ax.quiver(x_i, y_i, z_i, dy_i[0], dy_i[1], dy_i[2],
    arrow_length_ratio=0.1)
    ax.quiver(x_i, y_i, z_i, dz_i[0], dz_i[1], dz_i[2],
    arrow_length_ratio=0.1)
   
def verify_quat(final_traj, sphere_center, steps):
    print("----------- STRAT [verify_quat] -----------")
    for i in range(steps):
        if i % 5 == 0:
            print("final_traj[", i , "] = ", final_traj[i])
            translation_vector = final_traj[i][0]
            quaternion = final_traj[i][1]
            trafo = vector3D_quaternion_to_matrix(translation_vector, quaternion)
            print("trafo[", i , "] i to 0 = \n", trafo)

            trafo[:3, :3] = np.linalg.inv(trafo[:3, :3])
            trafo[:3, 3] = (-1)* trafo[:3, 3]
            print("trafo[", i , "] 0 to i = \n", trafo)

            vec_0 = sphere_center - translation_vector 
            # t_i = [0.0,0.0,0.0,1]
            # t_i = np.array(t_i)
            # t_i[:3] = vec_0

            vec = np.dot(trafo[:3, :3], vec_0)
            print("vec in coordinate 0: ", vec_0)
            print("vec in coordinate i: ", vec)
            print("--------------------------------------------------------")

def main():
    ## sample input 1
    sphere_center = [ -1.0, -1.0, 0.0]
    rotateAxis = [-0.2, -1.0, -1.0]
    start_point = [0.0, 0.6, 1.0]
    spanAngle = 210 

    ## sample input 2
    # sphere_center = [ -0.5, -0.5, 0.0]
    # start_point = [0.0, 1.0, 1.0]
    # rotateAxis = [0.0, 0.0, -1.0]
    # spanAngle = 99 

    # ## sample input 3
    # sphere_center = [ -0.5, -0.5, 0.2]
    # start_point = [0.8, -1.0, 1.0]
    # rotateAxis = [-1.0, -1.0, -1.0]
    # spanAngle = 270


    steps = 20
    quat = [1.0, 0.0, 0.0, 0.0]
    quat = np.array(quat)
    start_point = np.array(start_point), quat
    
    ## generate trajectory
    final_traj, circle_center, rotateAxis= \
        circular_trajectory(sphere_center, start_point, spanAngle, steps, rotateAxis)

    ## draw trajectory
    draw_trajectory( final_traj, circle_center, sphere_center, start_point, rotateAxis)
    
    ## verify that all the states are in positions with object(sphere center) at the same coordinate
    verify_quat(final_traj, sphere_center, steps)


if __name__ == "__main__":

    main()


