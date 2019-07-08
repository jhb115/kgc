import numpy as np

'''
This file contains models:
- create sorted list of (S, R, O) triplets
e.g. 
[S R O] =
[
[1, ..., ...] 
[1, ..., ...]
[1, ..., ...]
[2,
[3,
[3,
[3,
[4,
]
- create list where [ (start, end), (start, end), ... ]
e.g. [ (0, 3), (3, 4), (4, 7), ... ]  
'''
#%%%
# This function assumes that the reciprocal relationships (reciprocal triplets) already exist.

def sort_data(data):
    # Assume that the data is in array form above
    data.sort(axis=0)  # data gets sorted
    unique_ent = set(data[:, 0]).union( set(data[:, 2]))
    n_unique = len(unique_ent)
    # max_unique_ent = max(unique_ent)
    # if n_unique != max_unique_ent or n_unique != max_unique_ent:
    #     raise ValueError('n_unique is {} and max(unique_ent) is {}'.format(n_unique, max_unique_ent))

    i = 0
    curr_ent = data[0, 0]
    slice_dic = {}
    start = 0

    while i < len(data):

        prev_ent = curr_ent
        curr_ent = data[i, 0]

        if prev_ent != curr_ent:
            slice_dic[prev_ent] = (start, i)
            start = i

        if i == len(data) - 1:
            slice_dic[curr_ent] = (start, i+1)

        i += 1

    return data, slice_dic


def get_neighbor(sorted_data, subject, slice_dic):
    # we wish to find the neighbours of subject
    start_i, end_i = slice_dic[subject]

    return sorted_data[start_i:end_i]  # returns an array of neighbouring triplets

