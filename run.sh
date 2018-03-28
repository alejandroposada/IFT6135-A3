#!/usr/bin/env bash
if [ ! -d checkpoints ]; then
    mkdir "checkpoints"
fi

python3 train.py --controller_type "MLP" --cuda --M 128 --N 20 --controller_size 100 --learn_rate 1e-4