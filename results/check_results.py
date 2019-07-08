import matplotlib.pyplot as plt
import numpy as np
import os

print(os.getcwd())
print(os.listdir())

'''
Get ComplEx and ConvE results
'''


data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']

def load_data(name):
    hits = {}
    mrr = {}
    for each_data in data_list:
        folder_path = '{}/{}'.format(name, each_data)
        hits[each_data] = np.load('{}/train1/test_hit10.npy'.format(folder_path))
        mrr[each_data] = np.load('{}/train1/test_mrr.npy'.format(folder_path))

    return hits, mrr

# complex_hits, complex_mrr = load_data('ComplEx')
conve_n0_hits, conve_n0_mrr = load_data('ConvE_n0')
conve_n3_hits, conve_n3_mrr = load_data('ConvE_n3')

#%%%

for each_data in conve_n0_hits:
    print('ConvE (N0) for Dataset {}: MRR = {}, Hits@10 = {}'.format(each_data, str(conve_n0_mrr[each_data][-1]), str(conve_n0_hits[each_data][-1])))

print('')
print('')

for each_data in conve_n3_hits:
    print('ConvE (N3) for Dataset {}: MRR = {}, Hits@10 = {}'.format(each_data, str(conve_n3_mrr[each_data][-1]), str(conve_n3_hits[each_data][-1])))

print('')
print('For max')
print('')

for each_data in conve_n0_hits:
    print('ConvE (N0) for Dataset {}: MRR = {}, Hits@10 = {}'.format(each_data, str(max(conve_n0_mrr[each_data])), str(max(conve_n0_hits[each_data]))))

print('')
print('')

for each_data in conve_n3_hits:
    print('ConvE (N3) for Dataset {}: MRR = {}, Hits@10 = {}'.format(each_data, str(max(conve_n3_mrr[each_data])), str(max(conve_n3_hits[each_data]))))


#%%%


