#Generate train and test files for model training and cross validation,
#preprocessing the data provided in fullDataLabeled.txt beforehand.

from inspect import trace
import random
import math
import argparse
import sys

special_characters = ['"','#','$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~',',','.','!','?', '-', ''] #set of characters to remove

class GenerateFiles():
  def __init__(self, input, train_path, test_path):
    self.input = input #path to fullDataLabeled.txt
    self.train_path = train_path #path where trainMaster should be stored
    self.test_path = test_path #path where testMaster should be stored

  def concatenate(self, file1, file2, file3): #method to combine the datasets into one master file
    input1 = open(file1, 'r').read()
    input2 = open(file2, 'r').read()
    input3 = open(file3, 'r').read()

    fullDataOut = open("fulldataLabeled.txt", "wt") #method made to make fullDataLabeled

    fullDataOut.write(input1)
    fullDataOut.write(input2)
    fullDataOut.write(input3)

      
  def generate_test_list(self): #pick 400 numbers 1-3000 for generating test/training files
    test_list = []
    while len(test_list) < 400:
      temp = random.randrange(3000) + 1 #1-3000
      while temp in test_list:
        temp = random.randrange(3000) + 1 #1-3000
      test_list.append(temp)
    return test_list

  def generate_masters(self): #generate trainMaster.txt and testMaster.txt
      inputFile = open(self.input, 'r').readlines()
      trainOut = open(self.train_path, "wt")
      testOut = open(self.test_path, "wt")
      test_list = self.generate_test_list()
      for count, line in enumerate(inputFile):

        #lowercase the line
        line = line.lower()
        #remove special characters
        for i in special_characters:
          line = line.replace(i, '')

        if (count+1) in test_list:
          testOut.write(line)
        else:
          trainOut.write(line)
          

      trainOut.close()
      testOut.close()

  def generate_training(self): #set up the trainingSets directory and subdirectories for size 2600, 1300, and 650 lines each
    inputFile = open(self.train_path, 'r').readlines()

    for i in range (10):
      trainOut = open("./trainingSets/size2600TrainingSets/train" + str(i+1) + ".txt", "wt")
      #2600 loop
      for j in range (2600):
        temp = random.randrange(2600) #0-2599
        trainOut.write(inputFile[temp])
        

      trainOut = open("./trainingSets/size1300TrainingSets/train" + str(i+1) + ".txt", "wt")
      #1300 loop
      for j in range (1300):
        temp = random.randrange(2600) #0-2599
        trainOut.write(inputFile[temp])

      trainOut = open("./trainingSets/size650TrainingSets/train" + str(i+1) + ".txt", "wt")
      #650 loop
      for j in range (650):
        temp = random.randrange(2600) #0-2599
        trainOut.write(inputFile[temp])
    

def main(input, train, test ):
  file = GenerateFiles(input, train, test)
  file.generate_masters()
  file.generate_training()
  print("Files created")
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", help="Enter a path to the directory containing fulldataLabeled.")
  parser.add_argument("--train_path", help="Enter a path to the directory containing trainMaster.txt.")
  parser.add_argument("--test_path", help="Enter a path to the directory containing testMaster.txt.")
  args = parser.parse_args()  
  main(args.input_file, args.train_path, args.test_path)
