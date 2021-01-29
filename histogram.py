import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	df = pd.read_csv('datasets\dataset_train.csv')
	df.drop(['Index', 'First Name','Last Name','Birthday','Best Hand'], axis=1, inplace=True)
	custom_palette = {'Slytherin':'#55a868', 'Gryffindor':'#c44e52', 'Ravenclaw':'#4c72b0','Hufflepuff':'#FFF2B2'}
	sns.histplot(data=df, x='Care of Magical Creatures', hue='Hogwarts House', palette=custom_palette)
	plt.show()