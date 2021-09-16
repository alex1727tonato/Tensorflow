#!/bin/bash

p1=30
p2=40
cd codes
rm errorsB4.txt
python modelB-train.py $p1 $p2 2>> errorsB4.txt

