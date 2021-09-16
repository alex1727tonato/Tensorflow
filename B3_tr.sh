#!/bin/bash

p1=20
p2=30
cd codes
rm errorsB3.txt
python modelB-train.py $p1 $p2 2>> errorsB3.txt

