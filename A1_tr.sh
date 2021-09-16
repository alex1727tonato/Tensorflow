#!/bin/bash

p1=1
p2=10
cd codes
rm errorsA1.txt
python modelA-train.py $p1 $p2 2>> errorsA1.txt
