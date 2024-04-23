import numpy as np
import transforms3d


def vector3D_axangle_to_vector3D_quaternion(translation_vector, axangle):
    """ Converts a translation vector and a tuple of a 3 element sequence
    (vector specifying axis for rotation) and a scalar (angle of rotation) into
    a translation vector and a quaternion

    Args:
        translation_vector (list or numpy.ndarray): the translation vector with
        the shape=(3,)
        axangle (tuple): first component = the vector specifying axis for
        rotation (list or numpy.ndarray) with the shape=(3,); second component
        = angle of rotation (float)

    Returns:
        numpy.ndarray: the translation vector with the shape=(3,)
        numpy.ndarray: a quaternion with w, x, y and z component (shape=(4,))
    """

    # Get the components of the axis of rotation
    angle_axis = np.array(axangle[0])
    angle_rot = axangle[1]

    # Convert the rotation matrix into a quaternion
    quaternion = transforms3d.quaternions.axangle2quat(angle_axis, angle_rot)

    return translation_vector, quaternion
