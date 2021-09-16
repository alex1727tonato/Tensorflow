#!/bin/bash

P1=1
P2=50
cd codes
rm errors_blend.txt
python blend.py $P1 $P2  2>> errors_blend.txt

