#!/bin/bash

p1=10
p2=20
cd codes
rm errorsB2.txt
python modelB-train.py $p1 $p2 2>> errorsB2.txt

