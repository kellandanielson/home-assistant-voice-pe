#!/bin/bash
# Next retrain (when new recordings arrive): drop wavs into ~/mww/real_clips2|3 via score_new.py, then:
source ~/mww-venv/bin/activate
export LD_LIBRARY_PATH=$(find ~/mww-venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | paste -sd:)
SC=/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad
tr -d '\r' < $SC/train_hey_pickle_v9.py > ~/train_hey_pickle_v9.py
python ~/train_hey_pickle_v9.py data && python ~/train_hey_pickle_v9.py train v9_split v6_2 0.0001 2000   # honest gate: val = held-out
MWW_REAL_TRAIN=all python ~/train_hey_pickle_v9.py data && MWW_REAL_TRAIN=all python ~/train_hey_pickle_v9.py train v9_ship v6_2 0.0001 2000
echo V9NEXT-DONE
