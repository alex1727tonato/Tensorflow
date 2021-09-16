#!/bin/bash

p1=40
p2=50
cd codes
rm errorsA5.txt
python modelA-train.py $p1 $p2  2>> errorsA5.txt

