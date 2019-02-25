NOT_BROADCAST = 0
BROADCAST_1 = -1
BROADCAST_2 = 1
BROADCAST_1_SCALAR = -2
BROADCAST_2_SCALAR = 2
INVALID = 3


def is_broadcast(shape1, shape2):
    """
    Check whether this is a broadcast operation
    :param shape1: shape of the first operator
    :param shape2: shape of the second operator
    :return: 0 if not broadcast; -1 if the first has more dimension;
    1 if the second has more dimension; 2 if else
    """
    l1 = len(shape1)
    l2 = len(shape2)
    if shape1 == shape2:
        return NOT_BROADCAST
    elif l1 == l2 + 1 and shape1[1:] == shape2:
        return BROADCAST_1
    elif l2 == l1 + 1 and shape2[1:] == shape1:
        return BROADCAST_2
    elif l1 == 1 and shape1[0] == 1:
        return BROADCAST_2_SCALAR
    elif l2 == 1 and shape2[0] == 1:
        return BROADCAST_1_SCALAR
    else:
        return INVALID


def element_wise_binary(shape1, shape2):
    switch = {
        NOT_BROADCAST: (True, shape1),
        INVALID: (False, None),
        BROADCAST_1: (True, shape1),
        BROADCAST_2: (True, shape2),
        BROADCAST_1_SCALAR: (True, shape1),
        BROADCAST_2_SCALAR: (True, shape2)
    }

    re = is_broadcast(shape1, shape2)
    return switch[re]
