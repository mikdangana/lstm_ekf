PROJECT_DIR=./


all: clean run
	@echo "\nMake Done\n"


clean:
	@echo "\nCleaning...\n"
	rm --force --recursive *.pickle
	rm --force --recursive *.log

lint:
	@echo "\nLinting...\n"
	flake8 --exclude=.tox

test: clean
	@echo \n"Testing...\n"
	python ekf.py
	python lstm.py

run:
	@echo "\nRunning...\n"
	python lstm_ekf.py
