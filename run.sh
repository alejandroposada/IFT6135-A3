#!/usr/bin/env bash
if [ ! -d checkpoints ]; then
    mkdir "checkpoints"
fi

python3 train.py --controller_type "LSTM" --M 20 --N 128 --controller_size 100 --learn_rate 0.01