import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import argparse
import basic_math

def calc_stats(df_column):
	stats = [basic_math.my_count(df_column),
	basic_math.my_mean(df_column),
	basic_math.my_std(df_column),
	basic_math.my_min(df_column),
	basic_math.my_percentile(df_column, 0.25),
	basic_math.my_percentile(df_column, 0.5),
	basic_math.my_percentile(df_column, 0.75),
	basic_math.my_max(df_column)]
	return stats

def table(args):
	rows = {'name':[],'count':[],'mean':[],'std':[],'min':[],'25%':[],'50%':[],'75%':[],'max':[]}
	result = pd.DataFrame(rows) 
	return result

def describe(df, args):
	stats = {}
	for name_column in df.columns:
		if (name_column in args.include) or (args.all is True) or ((args.include == []) and
		(is_numeric_dtype(df[name_column])) and (name_column not in args.exclude)):
			col_stats = calc_stats(df[name_column])
			stats[name_column] = col_stats
	result = pd.DataFrame(stats, index =['count','mean','std','min','25%','50%','75%','max'])
	return result

def func_parser():
	my_parser = argparse.ArgumentParser(description='This is programm that describes a dataset')
	my_parser.add_argument('Path',metavar='path',type=str,help='the path to data')
	my_parser.add_argument('--all',action='store_true',help='show all columns')
	my_parser.add_argument('--include', action='store', type=str, nargs='*', default=[],
	help='show only selected columns')
	my_parser.add_argument('--exclude', action='store', type=str, nargs='*', default=[],
	help='don''t show selected columns')
	args = my_parser.parse_args()
	return args

def good_view(result_transform):
	print(result_transform)

if __name__ == '__main__':
	args = func_parser()
	input_path = args.Path
	try:
		df = pd.read_csv(input_path)
	except:
		exit(print('The file cannot be open'))
	try:
		good_view(describe(df, args))
	except:
		exit(print('There was unknown error'))