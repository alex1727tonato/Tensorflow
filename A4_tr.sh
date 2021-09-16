#!/bin/bash

p1=30
p2=40
cd codes
rm errorsA4.txt
python modelA-train.py $p1 $p2  2>> errorsA4.txt

