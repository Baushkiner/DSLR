import pandas as pd
import numpy as np
import argparse
import basic_math
from pandas.api.types import is_numeric_dtype
import csv

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def predict(classes, thetas, x):
	x = np.insert(x, 0, 1, axis=1)
	preds = [np.argmax(
		[sigmoid(xi @ theta) for theta in thetas]
	) for xi in x]
	return [classes[p] for p in preds]

def func_parser():
	my_parser = argparse.ArgumentParser(description='This is programm that describes a dataset')
	my_parser.add_argument('Path',metavar='path',type=str,help='the path to data')
	my_parser.add_argument('Weights',metavar='weights',type=str,help='the path to weights')
	args = my_parser.parse_args()
	return args

def score(classes, theta, x, y):
	return (predict(classes, theta, x) == y).mean()

if __name__ == '__main__':
	args = func_parser()
	input_path = args.Path
	try:
		dataset = pd.read_csv(input_path)
	except:
		exit(print('The file cannot be open'))
	input_weights = args.Weights
	try:
		columns = pd.read_csv(input_weights, nrows=1, header=None).values
		thetas = pd.read_csv(input_weights,skiprows=1, header=None).values.tolist()
	except:
		exit(print('The file cannot be open'))
	try:
		decode = {0:'Gryffindor', 1:'Hufflepuff', 2:'Ravenclaw', 3:'Slytherin'}
		df = dataset[columns[0]]
		df = df.fillna(0)
		t_norm = df.apply(lambda x: (x - basic_math.my_mean(x)) / basic_math.my_std(x) if x.name != 'Hogwarts House' else x)
		x = np.array(t_norm)
		y = np.array(dataset['Hogwarts House'].astype('category').cat.codes)
		classes = predict([0,1,2,3], thetas, x)
		for i in range(len(classes)):
			classes[i] = decode[classes[i]]
		answer = pd.DataFrame(classes)
		answer.to_csv('houses.csv', header=['Hogwarts House'],index_label='Index')
	except:
		exit(print('Unknown Error'))