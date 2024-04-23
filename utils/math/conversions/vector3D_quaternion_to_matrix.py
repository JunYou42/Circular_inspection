import numpy as np
import transforms3d


def vector3D_quaternion_to_matrix(translation_vector, quaternion):
    """ Converts a translation vector and a quaternion to a transformation
        matrix

    Args:
        translation_vector (list or numpy.ndarray): the translation vector with
        the shape=(3,)
        quaternion (list or numpy.ndarray): a quaternion with w, x, y and z
        component (shape=(4,))

    Returns:
        np.array: A transformation matrix with the shape=(4, 4)
    """

    # Convert the translation_vector into a numpy array
    translation_vector = np.array(translation_vector)

    # Convert the quaternion into a numpy array
    quaternion = np.array(quaternion)

    # Convert the quaternion into a rotation matrix
    rotation_matrix = transforms3d.quaternions.quat2mat(quaternion)

    # Create the trafo as an identiy matrix
    trafo = np.identity(4)

    # Transfer  the rotation matrix and the translation vector into the trafo
    trafo[:3, :3] = rotation_matrix
    trafo[:3, 3] = translation_vector

    return trafo
