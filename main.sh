#!/bin/bash

./B1_tr.sh &
./B2_tr.sh &
./B3_tr.sh &
./B4_tr.sh &
./B5_tr.sh &
./A1_tr.sh &
./A2_tr.sh &
./A3_tr.sh &
./A4_tr.sh &
./A5_tr.sh &
wait
./B_pr.sh &
./A_pr.sh &
wait
./blend.sh
