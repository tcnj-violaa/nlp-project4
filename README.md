This project was completed in Spring 2022 as the final project for CSC 427 - Natural Language Processing at TCNJ. It is being uploaded Summer 2022 for archival purposes.

This package implements two sentiment attribution models, normal Naive Bayes and Binary Naive Bayes, along with evaluation models F1-measure and Bootstrap.

We used b=1000 for our bootstrap sampling.

generateFiles.py: Source code for T1, T2, and T3, ie. generate train and test sets from the main data
To run, type:
$ python generateFiles.py --input_file=<path> --train_path=<train> --test_path=<test>
  <path> is path to fullDataLabeled.txt
  <train> is path to trainMaster.txt
  <test> is path to testMaster.txt

example: 
$ python generateFiles.py --input_file="./fulldataLabeled.txt" --train_path="./train/trainMaster.txt" --test_path="./test/testMaster.txt"

main.py: Source code for T4, T5, T6, T7, T8, and generating plots for T9

on elsa cluster, 
$ module add python/3.7.5
$ pip install matplotlib
for graphing functionality
to run main.py (example given below)
$ python main.py --train_files=<train> --test_path=<test>
  <train> is the path containing the three subdirectories
  <test> is the path to testMaster.txt

* Running main.py will create a new file called 'D3Output.txt'
This file is created to avoid overwriting the D3.txt file we used to create our plots. *

If you want to use the plot function, uncomment line 368 and pass in a valid 
input file with 1770 pairs of delta F1 and p-values, like D3.txt or D3Output.txt.

example: 
$ python main.py --train_files="./trainingSets" --test_path="./test/testMaster.txt"


plot1.pdf - plot from the 1770 pairs

plot2.pdf - plot from the 1770 pairs, with red O's meaning different data 
representations (like a count model vs. binary model) and blue X's (meaning count
model vs. count model or a binary model vs. a binary model)

D3.txt - contains the 1770 tuples, with delta F-measure and p-value

D4 - plot1.pdf and plot2.pdf

D5.txt - analysis of our results

D6.txt - responses to post project questions

