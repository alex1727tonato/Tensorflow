#!/bin/bash

p1=1
p2=10
cd codes
rm errorsB1.txt
python modelB-train.py $p1 $p2 2>> errorsB1.txt

