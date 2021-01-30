# DSLR
This is a School21's project that aims to implement logistic regression

# Usage
pip install .\requeirments.txt

### Reproduction of Pandas' function "describe"
python .\describe.py

### Graph analysis using graphs
python .\histogram.py

![alt text](https://github.com/Baushkiner/DSLR/blob/main/pictures/histogram.png)

python .\scatter_plot.py

![alt text](https://github.com/Baushkiner/DSLR/blob/main/pictures/scatter_plot.png)


python .\pair_plot.py

![alt text](https://github.com/Baushkiner/DSLR/blob/main/pictures/pair_plot.png)

### Reproduction LR - Train mode
##### Output - CSV file (weights.csv) with weights for each variable
python .\logreg_train.py datasets\dataset_train.csv

### Reproduction LR - Test mode
python .\logreg_predict.py datasets\dataset_test.csv  weights.csv

# Score
![alt text](https://github.com/Baushkiner/DSLR/blob/main/pictures/score.png)
