import torch
from kbc.models import Context_ComplEx
import numpy as np
from kbc.datasets import Dataset

dataset = Dataset('WN18RR')
sorted_data, slice_dic = dataset.get_sorted_train()


def test_complex_v1():
    nb_entities = dataset.get_shape()[0]
    nb_predicates = dataset.get_shape()[1]
    embedding_size = dataset.get_shape()[0]
    rs = np.random.RandomState(0)
    for _ in range(128):
        with torch.no_grad():
            model = Context_ComplEx(dataset.get_shape(), 200, sorted_data, slice_dic, max_NB=50, init_size=0.1,
                                    data_name='WN18RR',
                                    ascending=-1)
            xs = torch.from_numpy(rs.randint(nb_entities, size=32))
            xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
            xo = torch.from_numpy(rs.randint(nb_entities, size=32))
            scores, factors = model.forward(xp, xs, xo)
            inf = model.score(xp, xs, xo)
            scores_sp, scores_po = scores
            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()
            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)


#%%%