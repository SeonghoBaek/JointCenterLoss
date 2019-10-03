#!/bin/bash

#./clean.sh
#./augment.sh ./input/user/
#./align_for_model_training.sh
#python SR.py --mode train --model_path ./model_sr/m.ckpt --train_data output/aligned/user
python SR.py --mode train --model_path ./model_sr/m.ckpt --train_data input/user
