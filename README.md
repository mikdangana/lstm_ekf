# lstm_ekf

This is a research project studying the use of machine learning to tune Kalman filters. Measurement and process noise parameters for the extended Kalman filter (EKF) are typically curated by hand and distributed together with the filter once desirable performance is achieved during design testing. This approach produces tracking devices that work well enough for ideal conditions, but that can perform poorly for less optimal conditions, or unforeseen environments. 

In order to provide strong performance in all environments where the Kalman filter devices are used, a learning approach for parameter specification is better. This project attempts to accomplish and study that goal by using LSTM neural nets to estimate channel and noise covariance matrices. Hence whenever a Kalman filter device designed this way is deployed, it will first undergo a couple of minutes of tuning where these parameters are learned, after which the EKF runs independently for optimal tracking. 

Further calibration could be done on demand whenever device conditions are perceived to have changed, or if more tuning is considered desirable.

The EKF is used here to track system performance parameters and tune resource and configuration parameters via an LQN system model of the system. System parameters include operating system metrics like CPU utilization, disk time and utilization, number of processes, and system memory utilization. 

# Requirements

* Tensorflow, EKF python libraries
* Tested on Amazon AWS Deep Learning Image (Ubuntu) Version 5.0 (ami-c40c25bc)

# How to run

* pip install filterpy
* make

# Run all basic tests
* make test

# Run EKF tests
* make testekf

# Run LSTM tests
* make testltm

# Run controller tests
* make testctl

# Run controller convergence tests
* make testconvergence

# Note
* Read the Makefile for other tests and operations like cleaning, flushing, etc. Extend functionality as desired.
* Pickle files are generated when tests and/or the controller are ran. You may use src/plot.py to plot these

Feel free to email me if there are any issues. Enjoy!
