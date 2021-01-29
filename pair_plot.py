import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
	df = pd.read_csv('datasets\dataset_train.csv')
	df.drop(['Index', 'First Name','Last Name','Birthday','Best Hand'], axis=1, inplace=True)
	sns.set(font_scale=0.6)
	custom_palette = {'Slytherin':'#55a868', 'Gryffindor':'#c44e52', 'Ravenclaw':'#4c72b0','Hufflepuff':'#FFF2B2'}
	res = sns.pairplot(data = df, hue='Hogwarts House', palette=custom_palette,height=1, aspect=.8, plot_kws={'s':1.1})
	plt.show()