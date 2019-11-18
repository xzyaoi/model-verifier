#!/bin/bash

net=fc4
echo Evaluating network ${net}...
for spec in `ls ../test_cases/${net}`
do
	python verifier.py --net ${net} --spec ../test_cases/${net}/${spec}
done