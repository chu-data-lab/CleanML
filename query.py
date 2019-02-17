import pandas as pd
import config
import numpy as np
import utils

def load_rejects():
    two_tail_reject = pd.read_pickle('./table/t_test/two_tail_reject.pkl')
    one_tail_pos_reject = pd.read_pickle('./table/t_test/one_tail_pos_reject.pkl')
    one_tail_neg_reject = pd.read_pickle('./table/t_test/one_tail_neg_reject.pkl')
    rejects = {"two_tail": two_tail_reject, "one_tail_pos": one_tail_pos_reject, "one_tail_neg": one_tail_neg_reject}
    return rejects

def get_index_slice(fixed_dims, case, n_dims=6):
    idx = [slice(None, None, None) for i in range(n_dims)]
    for d, c in zip(fixed_dims, case):
        idx[d] = c
    return tuple(idx)

def get_domain(reject, dim):
    domain = list(set(reject.index.get_level_values(dim)))
    return domain

def query(reject, fixed_dims):
    queue = [[c] for c in get_domain(reject, fixed_dims[0])]
    result = {}
    cache = {}
    while len(queue) > 0:
        part_case = queue.pop(0)
        n_fix = len(part_case)
        part_dims = fixed_dims[0:n_fix]
        idx = get_index_slice(part_dims, part_case)
        part_reject = reject.loc[idx, :]
        count = "{}/{} ({:.0%})".format(np.sum(part_reject.values), len(part_reject.values), np.sum(part_reject.values)/len(part_reject.values))
        if n_fix == len(fixed_dims):
            result[tuple(part_case)] = count
        else:
            cache[tuple(part_case)] = count
            queue += [part_case + [c] for c in get_domain(part_reject, fixed_dims[n_fix])]
    return result, cache

def aggregate(rejects, fixed_dims, save_path):
    n_fixed = len(fixed_dims)
    test_types = sorted(list(rejects.keys()))[::-1]
    results = {}
    cache = {}
    for test_type, reject in rejects.items():
        results[test_type], cache[test_type] = query(reject, fixed_dims)  

    agg_result = {}
    for test_type in test_types:
        result = results[test_type]

        for key, value in result.items():
            new_key = []

            for i in range(n_fixed-1):
                new_key.append(key[i])
                part_key = key[0:i+1]
                new_key += [cache[t][part_key] for t in test_types]

            new_key.append(key[-1])
            new_key.append(test_type)

            agg_result[tuple(new_key)] = value   

    utils.dict_to_xls(agg_result, list(range(len(new_key)-1)), [len(new_key)-1], save_path)
    return agg_result


if __name__ == '__main__':
    rejects = load_rejects()
    q1 = aggregate(rejects, [0, 4], './table/query/q1.xls')
    q2 = aggregate(rejects, [0, 4, 2], './table/query/q2.xls')
    q3 = aggregate(rejects, [0, 4, 3], './table/query/q3.xls')
    q4 = aggregate(rejects, [0, 4, 2, 3], './table/query/q4.xls')
    q5 = aggregate(rejects, [0, 4, 2, 3, 1], './table/query/q5.xls')
    q6 = aggregate(rejects, [0, 4, 1], './table/query/q6.xls')

