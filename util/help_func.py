def merge_gradient(map1: dict, map2: dict)->dict:
    if len(map1) >= len(map2):
        for k, v in map2:
            if k in map1.keys():
                map1[k] += v
            else:
                map1[k] = v
        return map1
    else:
        for k, v in map1:
            if k in map2.keys():
                map2[k] += v
            else:
                map2[k] = v
        return map2
