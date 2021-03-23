# Addressing-the-Constraints-of-Active-Learning-on-the-Edge
Code used in the paper "Addressing the Constraints of Active Learning on the Edge" presented at PAISE 2021

# Directory structure
After running get_data.py (which gets and formats data), initial_model_train.py (trains initial models and makes data caches), and the experiments 
(bee_test.py, monkey_test.py, and mnist_test.py) which makes data in the folder results, the directory 
should look like the following.

```
.
├── AL
│   ├── __init__.py
│   └── algos.py
├── Data
│   ├── DataSets
│   │   ├── Bees
│   │   ├── MNIST
│   │   ├── Monkey
│   ├── __init__.py
│   └── data.py
├── Models
│   ├── bee_model.h5
│   ├── mnist_model.h5
│   └── monkey_model.h5
├── README.md
├── Zoo
│   ├── __init__.py
│   ├── beeCNN.py
│   ├── mnistCNN.py
│   ├── monkeyCNN.py
│   └── zoo.py
├── bee_test.py
├── engine.py
├── get_data.py
├── initial_model_train.py
├── mnist_test.py
├── monkey_test.py
├── requirements.txt
├── results
│   ├── bees
│   ├── mnist
│   └── monkey
└── tmp
    └── val_best_weights.h5

```

# Environment for code
Tested using python 3.6.12 and packages listed in requirements.txt

# How to get data
Data download links:
* MNIST: https://www.kaggle.com/oddrationale/mnist-in-csv
* Bees: https://www.kaggle.com/jenny18/honey-bee-annotated-images/home
* Monkey: https://www.kaggle.com/slothkong/10-monkey-species/home

Create a dir structure as such and place downloaded data in folders respectively:
```
.
├── DataSets
│   ├── Bees
│   │   └── archive.zip
│   ├── MNIST
│   │   └── archive.zip
│   └── Monkey
│       └── archive.zip
├── __init__.py
└── data.py
```
Then run data_prep.py. This will format the data in the folders and produce:
```
.
├── DataSets
│   ├── Bees
│   │   ├── archive.zip
│   │   └── raw_data
│   │       ├── data_tab.csv
│   │       └── images
│   ├── MNIST
│   │   ├── archive.zip
│   │   └── raw_data
│   │       ├── data_tab.csv
│   │       └── mnist.csv
│   └── Monkey
│       ├── archive.zip
│       └── raw_data
│           ├── data_tab.csv
│           └── images
├── __init__.py
└── data.py
```

# How to run experiments from paper
1. Run initial_train_model.py
2. Run mnist_test.py
3. Run bee_test.py
4. Run monkey_test.py
5. Run analyze_results.ipynb
