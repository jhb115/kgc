#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    command = f'PYTHONPATH=. python kbc/learn_grid.py --dataset FB237 ' \
        f'--model Context_ComplEx ' \
        f'--regularizer N4 ' \
        f'--max_epoch 100 ' \
        f'--mkdir 1 --rank {c["rank"]} --load_pre_train 1 --max_NB {c["max_NB"]} --valid 3 ' \
        f'--learning_rate 0.05 --reg 0.1 --batch_size 300 --g_weight {c["g_weight"]} --ascending -1'
    return command


def to_logfile(c, path):
    outfile = "{}/wn18rr_beaker_v1.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        rank=[100, 200, 400],
        max_NB=[10, 100],
        g_weight=[0, 0.03]
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/wn18rr/wn18rr_beaker_v1'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if is_rc is True and os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l tmem=9G
#$ -l h_rt=92:00:00
#$ -l gpu=1

date

source /share/apps/examples/python/python-3.6.5.source
source /share/apps/examples/cuda/cuda-9.0.source

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/jeunbyun/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jeunbyun/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jeunbyun/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jeunbyun/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate py36

date

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/jeunbyun/jeung_project
rm-rf kbc
git clone https://github.com/jhb115/kbc.git
cd kbc
python setup.py install
cd kbc/scripts
chmod +x download_data.sh
./download_data.sh
cd ../..
python kbc/process_datasets.py


""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
