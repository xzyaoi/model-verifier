for spec in `ls ../test_cases/fc1`
do
	python3 verifier.py --net fc1 --spec ../test_cases/fc1/${spec}
done