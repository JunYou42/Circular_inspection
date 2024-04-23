import numpy as np
import transforms3d


def vector3D_axangle_to_matrix(translation_vector, axangle):
    """ Converts a translation vector and a tuple of a 3 element sequence
    (vector specifying axis for rotation) and a scalar (angle of rotation) into
    a transformation matrix

    Args:
        translation_vector (list or numpy.ndarray): the translation vector with
        the shape=(3,)
        axangle (tuple): first component = the vector specifying axis for
        rotation (list or numpy.ndarray) with the shape=(3,); second component
        = angle of rotation (float)

    Returns:
        numpy.ndarray: A transformation matrix with the shape=(4, 4)
    """

    # Convert the translation_vector into a numpy array
    translation_vector = np.array(translation_vector)

    # Get the components from the axangle
    angle_axis = np.array(axangle[0])
    angle_rot = axangle[1]
    print(angle_axis)
    print(angle_rot)

    # Convert the axis of rotation into a rotation matrix
    rotation_matrix = transforms3d.axangles.axangle2mat(angle_axis, angle_rot)

    # Create the trafo as an identiy matrix
    trafo = np.identity(4)

    # Transfer  the rotation matrix and the translation vector into the trafo
    trafo[:3, :3] = rotation_matrix
    trafo[:3, 3] = translation_vector

    return trafo
