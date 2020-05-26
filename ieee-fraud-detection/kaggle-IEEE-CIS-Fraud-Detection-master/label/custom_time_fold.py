def time_kfold(length, idx, kfold=5):
    idxs = {}
    for i in range(1, kfold + 1):
        idxs["idx" + str(i)] = list(range(int(length * ((i - 1) / kfold)), int(length * ((i) / kfold))))

    nround = False
    if idx == 1:
        local_train_idx = idxs["idx2"] + idxs["idx3"] + idxs["idx4"] + idxs["idx5"]
        local_valid_idx = idxs["idx1"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 2:
        local_train_idx = idxs["idx1"] + idxs["idx3"] + idxs["idx4"] + idxs["idx5"]
        local_valid_idx = idxs["idx2"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 3:
        local_train_idx = idxs["idx1"] + idxs["idx2"] + idxs["idx4"] + idxs["idx5"]
        local_valid_idx = idxs["idx3"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 4:
        local_train_idx = idxs["idx1"] + idxs["idx2"] + idxs["idx3"] + idxs["idx5"]
        local_valid_idx = idxs["idx4"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 5:
        local_train_idx = idxs["idx3"] + idxs["idx4"] + idxs["idx5"]
        local_valid_idx = idxs["idx1"] + idxs["idx2"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 6:
        local_train_idx = idxs["idx2"] + idxs["idx4"] + idxs["idx5"]
        local_valid_idx = idxs["idx1"] + idxs["idx3"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 7:
        local_train_idx = idxs["idx2"] + idxs["idx3"] + idxs["idx5"]
        local_valid_idx = idxs["idx1"] + idxs["idx4"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 8:
        local_train_idx = idxs["idx1"] + idxs["idx4"] + idxs["idx5"]
        local_valid_idx = idxs["idx2"] + idxs["idx3"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 9:
        local_train_idx = idxs["idx1"] + idxs["idx3"] + idxs["idx5"]
        local_valid_idx = idxs["idx2"] + idxs["idx4"]
        train_idx = local_train_idx
        train_first = True
        train_second = False
    if idx == 10:
        local_train_idx = idxs["idx1"] + idxs["idx2"] + idxs["idx5"]
        local_valid_idx = idxs["idx3"] + idxs["idx4"]
        train_idx = local_train_idx
        train_first = True
        train_second = False

    return local_train_idx, local_valid_idx, train_idx, train_first, train_second, nround