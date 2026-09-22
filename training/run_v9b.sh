#!/bin/bash
source ~/mww-venv/bin/activate
export LD_LIBRARY_PATH=$(find ~/mww-venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | paste -sd:)
SC=/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad
until grep -q "V9-ALL-DONE" ~/mww/v9_stage.log; do sleep 20; done
tr -d '\r' < $SC/train_hey_pickle_v9.py > ~/train_hey_pickle_v9.py
MWW_REAL_W=2.0 MWW_NEG_CW=30 python ~/train_hey_pickle_v9.py train v9_c v6_2 0.0001 1500 || echo TRAIN-v9_c-FAILED
MWW_REAL_W=2.0 MWW_NEG_CW=40 python ~/train_hey_pickle_v9.py train v9_d v6_2 0.00005 3000 || echo TRAIN-v9_d-FAILED
MWW_REAL_W=3.0 MWW_NEG_CW=40 python ~/train_hey_pickle_v9.py train v9_e v6_2 0.0001 3000 || echo TRAIN-v9_e-FAILED
echo V9B-ALL-DONE
