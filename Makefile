PROJECT_DIR=./


all: clean run
	@echo "\nMake Done\n"


full: clean twod clean run
	@echo "\nMake Full Done\n"


clean:
	@echo "\nCleaning...\n"
	rm --force --recursive *.pickle
	rm --force --recursive *.log
	rm --force --recursive *.tar

lint:
	@echo "\nLinting...\n"
	flake8 --exclude=.tox


flush: clean
	@echo "\nRunning 2d...\n"
	python src/controller.py --iterations 1 --epochs 1 --twod  -t

monitor: clean
	@echo "\nRunning Monitors...\n"
	python src/controller.py

active: clean
	@echo "\nRunning Passive Monitors with Traffic...\n"
	python src/controller.py --generate-traffic

passive: clean
	@echo "\nRunning Passive Monitors with Traffic...\n"
	python src/controller.py --passive --generate-traffic

test: clean testekf testlstm testctl

testekf: clean
	@echo "\nTesting EKF Unit...\n"
	python src/ekf.py

testlstm: clean
	@echo "\nTesting LSTM Unit...\n"
	python src/lstm.py

testctl:
	@echo "\nTesting Control Unit...\n"
	python src/controller.py --iterations 10 --epochs 2 --test

run:
	@echo "\nRunning Controller...\n"
	python src/controller.py --iterations 10 --epochs 2 -t
