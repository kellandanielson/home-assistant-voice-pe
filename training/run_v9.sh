#!/bin/bash
source ~/mww-venv/bin/activate
export LD_LIBRARY_PATH=$(find ~/mww-venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | paste -sd:)
SC=/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad
tr -d '\r' < $SC/train_hey_pickle_v9.py > ~/train_hey_pickle_v9.py
python ~/train_hey_pickle_v9.py data || { echo V9-DATA-FAILED; exit 1; }
python ~/train_hey_pickle_v9.py train v9_ft62a v6_2 0.0003 3000 || echo TRAIN-v9_ft62a-FAILED
python ~/train_hey_pickle_v9.py train v9_ft62b v6_2 0.0001 3000 || echo TRAIN-v9_ft62b-FAILED
python ~/train_hey_pickle_v9.py train v9_ft56a v5_6 0.0003 3000 || echo TRAIN-v9_ft56a-FAILED
python ~/train_hey_pickle_v9.py train v9_s1 scratch 0.001 10000 || echo TRAIN-v9_s1-FAILED
echo V9-ALL-DONE
