#!/bin/bash

rm -rf sr
python SR.py --mode test --model_path ./model_sr/m.ckpt --test_data gan
