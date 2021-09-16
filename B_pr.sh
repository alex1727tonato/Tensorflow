#!/bin/bash

p1=1
p2=50
cd codes
rm errorsB_pred.txt
python modelB-pred.py $p1 $p2 2>> errorsB_pred.txt

