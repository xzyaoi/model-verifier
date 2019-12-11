for spec in `ls ../test_cases/fc4`
do
	python3 verifier.py --net fc4 --spec ../test_cases/fc4/${spec}
done