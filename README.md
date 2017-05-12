# Rosetta-learn

Rosetta-learn is a recommender system for protein sequence optimization using Rosetta. Users who want to model their high-througput protein sequencing results and associated Rosetta metrics can do so easily with Rosetta-learn. 

Rosetta-learn builds and tunes an optimal deep neural network (DNN) to model protein data. Using this model, Rosetta-learn recommends an optimized sequence - predicted to maximize experimental output. 

## Installation
	pip install rosetta-learn

## Usage
Rosetta-learn requires an input xlsx file of protein sequencing data and their respecitve Rosetta metrics with the following structure:

Example:

| Sequences    | Rosetta Metric 1 | Rosetta Metric 2 | ... | Output        |
| -------------|----------------- |------------------|-----|:-------------:|
| actgactg ... |     12           |   4              | ... | 3      |
| actgactg ... |      3           |    8.3           | ... | 5      |



To generate a new model using the command line interface:

	rosetta-learn input.xlsx

To retrain a previously generated model using the command line interface:

	rosetta-learn input.xlsx -m model.h5
