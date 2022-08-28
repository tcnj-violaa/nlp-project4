## This project was completed in Spring 2022 as the final project for CSC 427 - Natural Language Processing at TCNJ. It is being uploaded Summer 2022 for archival purposes.

This package implements two sentiment attribution models, normal Naive Bayes and Binary Naive Bayes, along with evaluation models F1-measure and bootstrap. This project uses Python 3.7.5.

## File descriptions
* `generateFiles.py`: Source code for preprocessing the data stored in `fullDataLabeled.txt` and generating train and test sets from it.

Usage:
`$ python generateFiles.py --input_file=<path> --train_path=<train> --test_path=<test>`

  * `<path>` is the path to fullDataLabeled.txt
  * `<train>` is the path to trainMaster.txt
  * `<test>` is the path to testMaster.txt

Example: 
`$ python generateFiles.py --input_file="./fulldataLabeled.txt" --train_path="./train/trainMaster.txt" --test_path="./test/testMaster.txt"`

Note: We used b=1000 for our bootstrap sampling.

 ---
 
* `main.py`: Source code for implementing, training, and running normal/binary Naive Bayes Models and generating plots for analysis.

Running main.py will create a new file called 'D3Output.txt'
This file is created to avoid overwriting the D3.txt file we used to create our plots. *

If you want to use the plot function, uncomment line 368 and pass in a valid 
input file with 1770 pairs of delta F1 and p-values, like D3.txt or D3Output.txt or model-comparison-cross-validation-results.txt

Usage: 
`$ python main.py --train_files="./trainingSets" --test_path="./test/testMaster.txt"`

---

* `plot1.pdf` - plot from the 1770 pairs comparing binary Naive Bayes and standard (count the # of occurrences of each word) Naive Bayes

* `plot2.pdf` - plot from the 1770 pairs, with red O's meaning different data 
representations (like a count model vs. binary model) and blue X's (meaning count
model vs. count model or a binary model vs. a binary model)

* `model-comparison-cross-validation-results.txt` - contains 1770 tuples, comparing models with delta F-measure and p-value

* `analysis.txt` - analysis of results found in `plot1.pdf` and `plot2.pdf`

## Running on the TCNJ HPC Cluster
To run on the cluster, first run:
`$ module add python/3.7.5`
to ensure the correct python version is being used,
and 
`$ pip install matplotlib`
for graphing functionality. 



