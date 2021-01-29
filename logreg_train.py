import pandas as pd
import numpy as np
import argparse
import basic_math
from pandas.api.types import is_numeric_dtype
import csv
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(theta, x, y):
	h = sigmoid(x @ theta)
	m = len(y)
	cost = 1 / m * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
	grad = 1 / m * ((y - h) @ x)
	return cost, grad

def fit(x, y, max_iter=1000, alpha=0.1):
	x = np.insert(x, 0, 1, axis=1)
	thetas = []
	classes = np.unique(y)
	costs = np.zeros(max_iter)

	for c in classes:
		# one vs. rest binary classification
		binary_y = np.where(y == c, 1, 0)
		
		theta = np.zeros(x.shape[1])
		for epoch in range(max_iter):
			costs[epoch], grad = cost(theta, x, binary_y)
			theta += alpha * grad
		thetas.append(theta)
	return thetas, classes, costs

def predict(classes, thetas, x):
	x = np.insert(x, 0, 1, axis=1)
	preds = [np.argmax(
		[sigmoid(xi @ theta) for theta in thetas]
	) for xi in x]
	return [classes[p] for p in preds]

def score(classes, theta, x, y):
	return (predict(classes, theta, x) == y).mean()


def func_parser():
	my_parser = argparse.ArgumentParser(description='This is programm that describes a dataset')
	my_parser.add_argument('Path',metavar='path',type=str,help='the path to data')
	my_parser.add_argument('--columns', action='store', type=str, nargs='*',default=['Herbology',
	'Ancient Runes', 'Astronomy', 'Charms'],
	help='Train by selected columns, default [Herbology, Ancient Runes, Astronomy, Charms]')
	my_parser.add_argument('--max_iter', action='store', type=int, nargs='?', default=300,
	help='Max_iter value must be more than 0, default = 300')
	my_parser.add_argument('--alpha', action='store', type=float, nargs='?',default=0.1,
	help='Alpha value must be between 0 and 1, default = 0.1')
	args = my_parser.parse_args()
	return args

def prepare_df(dataset, args):
	for column_name in args.columns:
		if column_name == 'Hogwarts House':
			exit(print('Error, the same are target and one of columns', column_name, sep='	'))
		if column_name not in dataset.columns:
			exit(print('Error, there is no the column in data', column_name, sep='	'))
		if not is_numeric_dtype(dataset[column_name]):
			exit(print('Error, there is no the numeric column', column_name, sep='	'))
	return

def train(dataset, args):
	prepare_df(dataset, args)
	df = dataset[args.columns]
	df = df.fillna(0)

	df['Hogwarts House'] = dataset['Hogwarts House'].astype('category').cat.codes

	t_norm = df.apply(lambda x: (x - basic_math.my_mean(x)) / basic_math.my_std(x) if x.name != 'Hogwarts House' else x)

	data = np.array(t_norm)
	x_train, y_train = data[:, :-1], data[:, -1]

	if (args.alpha > 1) or (args.alpha < 0):
		args.alpha = 0.1
	if (args.max_iter < 0) or (args.max_iter > 10000):
		args.max_iter = 300
	thetas, classes, costs = fit(x_train, y_train,args.max_iter, args.alpha)
	plt.plot(costs)
	plt.xlabel('Number Epochs')
	plt.ylabel('Cost')
	plt.show()
	print(f"Train Accuracy: {score(classes, thetas, x_train, y_train):.3f}")

	with open('weights.csv', 'w', newline="") as f:
		write = csv.writer(f)
		write.writerow(args.columns)
		write.writerows(thetas)
	return

if __name__ == '__main__':
	args = func_parser()
	input_path = args.Path
	try:
		dataset = pd.read_csv(input_path)
	except:
		exit(print('The file cannot be open'))
	try:
		train(dataset, args)
	except:
		exit(print('The error in train'))