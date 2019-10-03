#!/bin/bash

python JointCenterLoss.py --mode fpr --model_path ./model/m.ckpt --test_data ./Unknown --train_data output/aligned/user --align True
