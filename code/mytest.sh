for spec in `ls ../test_cases/fc5`
do
	python3 verifier.py --net fc5 --spec ../test_cases/fc5/${spec}
done