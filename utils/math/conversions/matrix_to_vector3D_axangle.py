import numpy as np
import transforms3d


def matrix_to_vector3D_axangle(trafo):
    """ Converts a transformation matrix into a translation
    vector and a tuple of a 3 element sequence (vector specifying axis for
    rotation) and a scalar (angle of rotation).

    Args:
        trafo (list or numpy.ndarray): A transformation matrix with the
            shape=(4, 4) or shape=(3, 4)

    Returns:
        numpy.ndarray: the translation vector with the shape=(3,)
        tuple: first component = vector specifying axis for rotation with
        shape=(3,); second component = angle of rotation (float)
    """

    # Convert the trafo into a numpy array
    trafo = np.array(trafo)

    # The shape of the trafo only matters if there are less components then
    # (3,4)
    rotation_matrix = trafo[:3, :3]
    translation_vector = trafo[:3, 3]

    # Convert the rotation matrix into axis of rotation
    angle_axis, angle_rot = transforms3d.axangles.mat2axangle(rotation_matrix)

    return translation_vector, (angle_axis, angle_rot)
