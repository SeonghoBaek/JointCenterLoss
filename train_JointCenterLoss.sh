#!/bin/bash

#./clean.sh
#./augment.sh ./input/user/
#./align_for_model_training.sh
python JointCenterLoss.py --mode train --model_path ./model/m.ckpt --train_data output/aligned/user
