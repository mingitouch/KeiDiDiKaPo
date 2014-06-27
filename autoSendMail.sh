#!/bin/sh

echo "run generate.py"
python generate.py
mail -s "Your program has finished" mingitouch@gmail.com < nohup.out
