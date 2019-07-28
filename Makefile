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
	rm --force --recursive *.train
	rm --force --recursive *.state


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
	@echo "\nRunning Active Monitors with Traffic...\n"
	python src/controller.py --generate-traffic --n_samples 70
	reset


passive: clean
	@echo "\nRunning Passive Monitors with Traffic...\n"
	python src/controller.py --passive --generate-traffic
	reset


cleanremote:
	python src/controller.py --test-clean --search "{1..3}" --replace "{1..$(iter)}"


steps: clean cleanremote
	python src/controller.py --generate-traffic --n_samples 70 --search "{1..3}" --replace "{1..$(iter)}"
	rm --force --recursive run_$(iter)
	mkdir run_$(iter)
	mv *.tar run_$(iter)
	@echo "\nrun_$(iter) done\n"
	reset


stats:  clean active cleanremote
	rm --force --recursive run_$(iter)
	mkdir run_$(iter)
	mv *.tar run_$(iter)
	@echo "\nrun_$(iter) done\n"
	reset


test: clean testekf testlstm testctl testconvergence testmodeltrack


testekf: clean
	@echo "\nTesting EKF Unit...\n"
	python src/ekf.py


testlstm: clean
	@echo "\nTesting LSTM Unit...\n"
	python src/lstm.py


testctl: clean
	@echo "\nTesting Control Unit...\n"
	python src/controller.py --iterations 10 --epochs 2 --test


testconvergence: clean
	@echo "\nTesting LSTM-EKF Convergence...\n"
	python src/controller.py --test-convergence 0
	python src/controller.py --test-convergence 1
	python src/controller.py --test-convergence 2
	python src/controller.py --test-convergence 3
	python src/controller.py --test-convergence 4
	python src/controller.py --test-convergence 5


testmodeltrack: clean
	@echo "\nTesting LSTM LQN Model Tracking...\n"
	python src/controller.py --track-model --n_msmt 7 --n_lstm_out 7 --n_entries 1


run:
	@echo "\nRunning Controller...\n"
	python src/controller.py --iterations 10 --epochs 2 -t
