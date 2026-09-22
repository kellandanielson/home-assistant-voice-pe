#!/bin/bash
source ~/mww-venv/bin/activate
export LD_LIBRARY_PATH=$(find ~/mww-venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | paste -sd:)
SC=/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad
tr -d '\r' < $SC/train_hey_pickle_v9.py > ~/train_hey_pickle_v9.py
export MWW_REAL_TRAIN=real_clips_train
python ~/train_hey_pickle_v9.py data || { echo V9F-DATA-FAILED; exit 1; }
python ~/train_hey_pickle_v9.py train v9_f1 v6_2 0.0001 3000 || echo TRAIN-v9_f1-FAILED
python ~/train_hey_pickle_v9.py train v9_f2 v6_2 0.0001 2000 || echo TRAIN-v9_f2-FAILED
echo V9F-ALL-DONE
