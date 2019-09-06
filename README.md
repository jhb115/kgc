## Installation
Create a conda environment with pytorch cython and scikit-learn :
```
conda create --name kbc_env python=3.7
source activate kbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the kbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are download, add them to the package data folder by running :
```
python kbc/process_datasets.py
```

This will create the files required to compute the filtered metrics.

