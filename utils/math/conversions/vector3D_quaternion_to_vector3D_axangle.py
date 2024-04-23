import transforms3d


def vector3D_quaternion_to_vector3D_axangle(translation_vector, quaternion):
    """ Converts a translation vector and a quaternion into a translation
    vector and a tuple of a 3 element sequence (vector specifying axis for
    rotation) and a scalar (angle of rotation).

    Args:
        translation_vector (list or numpy.ndarray): the translation vector with
        the shape=(3,)
        quaternion (list or numpy.ndarray): a quaternion with w, x, y and z
        component (shape=(4,))

    Returns:
        numpy.ndarray: the translation vector with the shape=(3,)
        tuple: first component = vector specifying axis for rotation with
        shape=(3,); second component = angle of rotation (float)
    """

    # Convert the rotation matrix into a quaternion
    angle_axis, angle_rot = transforms3d.quaternions.quat2axangle(quaternion)

    return translation_vector, (angle_axis, angle_rot)
