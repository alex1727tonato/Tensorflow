#!/bin/bash

p1=20
p2=30
cd codes
rm errorsA3.txt
python modelA-train.py $p1 $p2  2>> errorsA3.txt

