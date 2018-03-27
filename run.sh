#!/usr/bin/env bash
if [ ! -d checkpoints ]; then
    mkdir "checkpoints"
fi

python3 train.py