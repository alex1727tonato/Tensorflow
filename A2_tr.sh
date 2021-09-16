#!/bin/bash

p1=10
p2=20
cd codes
rm errorsA2.txt
python modelA-train.py $p1 $p2  2>> errorsA2.txt

