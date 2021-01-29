import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
	df = pd.read_csv('datasets\dataset_train.csv')
	custom_palette = {'Slytherin':'#55a868', 'Gryffindor':'#c44e52', 'Ravenclaw':'#4c72b0','Hufflepuff':'#FFF2B2'}
	sns.scatterplot(x=df['Defense Against the Dark Arts'], y=df['Astronomy'], hue=df['Hogwarts House'], palette=custom_palette)
	plt.show()