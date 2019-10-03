#!/bin/bash

rm -rf gan
python FDN.py --mode test --model_path ./model_aae/m.ckpt --test_data test_data
