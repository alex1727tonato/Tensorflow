#!/bin/bash

p1=1
p2=50
cd codes
rm errorsA_pred.txt
python modelA-pred.py $p1 $p2 2>> errorsA_pred.txt

