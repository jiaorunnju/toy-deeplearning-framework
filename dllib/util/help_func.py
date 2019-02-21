def merge_gradient(map1: dict, map2: dict)->dict:
    """
    Add the gradient of a variable appears in 2 operations
    :param map1: output of backward of first operation
    :param map2: output of backward of second operation
    :return: merged output
    """
    if len(map1) >= len(map2):
        for k, v in map2.items():
            if k in map1.keys():
                map1[k] += v
            else:
                map1[k] = v
        return map1
    else:
        for k, v in map1.items():
            if k in map2.keys():
                map2[k] += v
            else:
                map2[k] = v
        return map2
