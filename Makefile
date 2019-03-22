PROJECT_DIR=./


all: clean run
	@echo "\nMake Done\n"


full: clean twod clean run
	@echo "\nMake Full Done\n"


clean:
	@echo "\nCleaning...\n"
	rm --force --recursive *.pickle
	rm --force --recursive *.log

lint:
	@echo "\nLinting...\n"
	flake8 --exclude=.tox


flush: clean
	@echo "\nRunning 2d...\n"
	python src/controller.py --iterations 1 --epochs 1 --twod 

test: clean
	@echo \n"Testing...\n"
	python src/ekf.py
	python src/lstm.py

run:
	@echo "\nRunning...\n"
	python src/controller.py --iterations 100 --epochs 4
