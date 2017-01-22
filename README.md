# Machine Learning Repository

This repository is an playground to implement any type of machine learning algorihtm.

**Currently Implemented Algorithms:**

## Evolutionary Algorithms
* Genetic Programming (Standard Version)

## Neural Networks
* Restricted Boltzmann Machine with dynamic numbers of hidden neurons
* Single Layer Perceptron (binary classification, multinominal classification has to be done with a one vs all strategy)
* Multi Layer Perceptron (binary and multinominal classification)

## Deep Architectures
* Deep Belief Network (pretrained rbms stacked into a mlp)


## Development Notes
* install python requirements in virtualenvironment based on `requirements.txt`.
* docs are created by running `make html` inside `doc/` directory
* tests are run by executing `nosetests` in the root of the project (use `nosetests --nocapture` to also see print statements, to run a single test use e.g. `nosetests path.to.module:Class.test_method`)
