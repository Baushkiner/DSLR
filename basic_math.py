import numpy as np
from math import sqrt
from math import floor
from math import ceil

def my_count(df_column):
	try:
		counter = 0
		for value in df_column:
			if not np.isnan(value):
				counter += 1
		return counter
	except:
		return np.nan

def my_mean(df_column):
	try:
		value_sum = 0
		for value in df_column:
			if not np.isnan(value):
				value_sum += value
		counter = my_count(df_column)
		return value_sum / counter
	except:
		return np.nan

def my_std(df_column):
	try:
		value_mean = my_mean(df_column)
		distance_sum = 0
		for value in df_column:
			if not np.isnan(value):
				distance_sum += (value_mean - value) * (value_mean - value)
		counter = my_count(df_column)
		value_std = sqrt(distance_sum / counter)
		return value_std
	except:
		return np.nan

def my_min(df_column):
	try:
		value_min = np.nan
		for value in df_column:
			if not np.isnan(value):
				if np.isnan(value_min):
					value_min = value
				elif value_min > value:
					value_min = value
		return value_min
	except:
		return np.nan

def my_max(df_column):
	try:
		value_max = np.nan
		for value in df_column:
			if not np.isnan(value):
				if np.isnan(value_max):
					value_max = value
				elif value_max < value:
					value_max = value
		return value_max
	except:
		return np.nan

def my_percentile(df_column, percent):
	try:
		N = []
		for value in df_column:
			if not np.isnan(value):
				N.append(value)
		N.sort()
		k = (len(N) - 1) * percent
		f = floor(k)
		c = ceil(k)
		if f == c:
			return N[int(k)]
		d0 = N[int(f)] * (c - k)
		d1 = N[int(c)] * (k - f)
		return d0 + d1
	except:
		return np.nan