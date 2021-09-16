#!/bin/bash

p1=40
p2=50
cd codes
rm errorsB5.txt
python modelB-train.py $p1 $p2 2>> errorsB5.txt

