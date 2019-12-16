#!/bin/bash

net=conv1
echo Evaluating network ${net}...
for spec in `ls ../test_cases/${net}`
do
	python3 verifier.py --net ${net} --spec ../test_cases/${net}/${spec}
done