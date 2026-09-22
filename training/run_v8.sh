#!/bin/bash
source ~/mww-venv/bin/activate
export LD_LIBRARY_PATH=$(find ~/mww-venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | paste -sd:)
SC=/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad
tr -d '\r' < $SC/train_hey_pickle_v8.py > ~/train_hey_pickle_v8.py
python ~/train_hey_pickle_v8.py data || { echo V8-DATA-FAILED; exit 1; }
for n in v8_1 v8_2 v8_3; do python ~/train_hey_pickle_v8.py train $n || echo TRAIN-$n-FAILED; done
echo V8-ALL-DONE
