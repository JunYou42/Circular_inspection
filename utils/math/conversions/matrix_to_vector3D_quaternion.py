import numpy as np
import transforms3d


def matrix_to_vector3D_quaternion(trafo):
    """ Converts a transformation matrix into a translation vector and a
        quaternion

    Args:
        trafo (list or numpy.ndarray): A transformation matrix with the
            shape=(4, 4) or shape=(3, 4)

    Returns:
        np.array: the translation vector with the shape=(3,)
        np.array: a quaternion with w, x, y and z component (shape=(4,))
    """

    # Convert the trafo into a numpy array
    trafo = np.array(trafo)

    # The shape of the trafo only matters if there are less components then
    # (3,4)
    rotation_matrix = trafo[:3, :3]
    translation_vector = trafo[:3, 3]

    # Convert the rotation matrix into a quaternion
    quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)

    return translation_vector, quaternion
