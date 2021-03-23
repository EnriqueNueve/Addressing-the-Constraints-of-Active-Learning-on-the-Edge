# Addressing-the-Constraints-of-Active-Learning-on-the-Edge
Code used in the paper "Addressing the Constraints of Active Learning on the Edge" presented at PAISE 2021

# Directory structure with data and models
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

# How to run experiments from paper
1. Run get_data.py
2. Run initial_train_model.py
3. Run mnist_test.py
4. Run bee_test.py
5. Run monkey_test.py
6. Run analyze_results.py
