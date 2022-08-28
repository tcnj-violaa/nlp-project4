# Implement models and training routines, as well as the generation of
# graphs for comparative analysis. 
import random
import argparse
from matplotlib import pyplot as plt

class NBCount: 
  #initialize the Naive Bayes count model
  def __init__ (self, sentences):
    self.sentences = sentences # current training document to read
    self.posWordCount = 0 # total word count for positive reviews
    self.negWordCount = 0 # total word count for negative reviews
    self.nbPosCount = {} # individual token coutns for positive reviews
    self.nbNegCount = {} # individual token counts for negative reviews
    self.nbPosProbs = {} # token probabilities for positive reviews
    self.nbNegProbs = {} # token probabilities for negative reviews
    self.posReviewProb = 0 # percentage of positive reviews in the training sets
    self.negReviewProb = 0 # percentage of negative reviews in the traiing sets
    self.predictions = {} # store predictions for faster bootstrapping
    self.truth = {} # store truths for faster bootstrapping

   
    #get count of each unigram

    negReviewCount = 0 # total word count for all negative reviews
    posReviewCount = 0 # total word count for all positive reviews

    # iterate through all the reviews in the document.
    # if a review is negative, update the counts for each token in the negative words dictionary
    # otherwise the review is positive, update the counts for each token in the positive words dictionary
    for line in self.sentences:
      if line[-2] == "0":
        negReviewCount += 1
        line = line[:-2]
        splitLine = line.split()
        for word in splitLine:
          self.negWordCount += 1
          if word in self.nbNegCount:
            self.nbNegCount[word] = self.nbNegCount.get(word) + 1
          else:      
            self.nbNegCount[word] = 1
      else:
        posReviewCount += 1
        line = line[:-2]
        splitLine = line.split()
        for word in splitLine:
          self.posWordCount += 1
          if word in self.nbPosCount:
            self.nbPosCount[word] = self.nbPosCount.get(word) + 1
          else:      
            self.nbPosCount[word] = 1   

    # calculate occurrance percentages
    self.posReviewProb = posReviewCount / (posReviewCount + negReviewCount)
    self.negReviewProb = 1 - self.posReviewProb

    # merge the two dictionaries for positive and negative word tokens into one
    mergedDict = self.nbPosCount
    for word in self.nbNegCount:
      if word not in mergedDict:
        mergedDict[word] = 1 # value does not matter in this case
        
    # record the number of unique tokens in the vocabulary
    lenMerged = len(mergedDict)

    # for each word in the merged dictionary
    # if the word exists in the positive token dictionary, then use the pre-existing
    # probability for that word and add 1 for smoothing
    # otherwise, the word never appears in a positive review. The probability
    # should be 1 due to add 1 smoothing
    for word in mergedDict: 
      if word in self.nbPosCount:
        self.nbPosProbs[word] = (self.nbPosCount[word] + 1) / (self.posWordCount + lenMerged)
      else: 
        self.nbPosProbs[word] = 1 / (self.posWordCount + lenMerged)

    # for each word in the merged dictionary
    # if the word exists in the negative token dictionary, then use the pre-existing
    # probability for that word and add 1 for smoothing to calculate the new probability
    # otherwise, the word never appears in a negative review. The numerator
    # should be 1 due to add 1 smoothing
    for word in mergedDict: 
      if word in self.nbNegCount:
        self.nbNegProbs[word] = (self.nbNegCount[word] + 1) / (self.negWordCount + lenMerged)        
      else:
        self.nbNegProbs[word] = 1 / (self.negWordCount + lenMerged)


class NBBinary: 
  #initialize the Naive Bayes Binary model
  def __init__ (self, sentences):
    self.sentences = sentences # current training document to read
    self.posWordCount = 0 # total word count for positive reviews
    self.negWordCount = 0 # total word count for negative reviews
    self.nbPosCount = {} # individual token coutns for positive reviews
    self.nbNegCount = {} # individual token counts for negative reviews
    self.nbPosProbs = {} # token probabilities for positive reviews
    self.nbNegProbs = {} # token probabilities for negative reviews
    self.posReviewProb = 0 # percentage of positive reviews in the training sets
    self.negReviewProb = 0 # percentage of negative reviews in the traiing sets
    self.predictions = {} # store predictions for faster bootstrapping
    self.truth = {} # store truths for faster bootstrapping

    negReviewCount = 0 # total word count for all negative reviews
    posReviewCount = 0 # total word count for all positive reviews

    # iterate through all the reviews in the document.
    # create a set using the unique words of the current review
    # if a review is negative, update the word count if the encountered word is unique
    # otherwise the review is positive, update the counts for each unique token in the positive words dictionary
    for line in self.sentences:
      if line[-2] == "0":
        negReviewCount += 1
        line = line[:-2]
        splitLine = line.split()
        uniqueWords = set(splitLine)
        for word in uniqueWords:
          self.negWordCount += 1
          if word in self.nbNegCount:
            self.nbNegCount[word] = self.nbNegCount.get(word) + 1
          else:      
            self.nbNegCount[word] = 1
      else: #positive
        posReviewCount += 1
        line = line[:-2]
        splitLine = line.split()
        uniqueWords = set(splitLine)
        for word in uniqueWords:
          self.posWordCount += 1
          if word in self.nbPosCount:
            self.nbPosCount[word] = self.nbPosCount.get(word) + 1
          else:      
            self.nbPosCount[word] = 1   

    # calculate occurrance percentages
    self.posReviewProb = posReviewCount / (posReviewCount + negReviewCount)
    self.negReviewProb = 1 - self.posReviewProb

    # merge the two dictionaries for positive and negative token counts into one
    mergedDict = self.nbPosCount
    for word in self.nbNegCount:
      if word not in mergedDict:
        mergedDict[word] = 1 #value does not matter in this case

    # record the number of unique tokens in the vocabulary
    lenMerged = len(mergedDict)

    # for each word in the merged dictionary
    # if the word exists in the positive token dictionary, then use the pre-existing
    # probability for that word and add 1 for smoothing
    # otherwise, the word never appears in a positive review. The probability
    # should be 1 due to add 1 smoothing
    for word in mergedDict: 
      if word in self.nbPosCount:
        self.nbPosProbs[word] = (self.nbPosCount[word] + 1) / (self.posWordCount + lenMerged)
      else:
        self.nbPosProbs[word] = 1 / (self.posWordCount + lenMerged)

    # for each word in the merged dictionary
    # if the word exists in the negative token dictionary, then use the pre-existing
    # probability for that word and add 1 for smoothing to calculate the new probability
    # otherwise, the word never appears in a negative review. The numerator
    # should be 1 due to add 1 smoothing
    for word in mergedDict: 
      if word in self.nbNegCount:
        self.nbNegProbs[word] = (self.nbNegCount[word] + 1) / (self.negWordCount + lenMerged)        
      else:
        self.nbNegProbs[word] = 1 / (self.negWordCount + lenMerged)


def populateProbs(model, testFile):
  # parameters: 
    #model = NBCount or NBBinary model
    #testFile = list representation of the test file
  
  test = testFile
  index = 0
  for line in test: #loop through each line in the testFile
    truth = line[-2] #ground truth of the review
    line = line[:-2]
    model.truth[index] = truth
    splitLine = line.split()
    posLikelihood = model.posReviewProb #probability the review is positive
    negLikelihood = model.negReviewProb #probability the review is negative
    for word in splitLine:
      #skip unknown words; posProbs and negProbs should have the same vocab 
      #since we're using add-one smoothing
      if word not in model.nbPosProbs:
        continue
      else: #multiply by the probablity of the word
        posLikelihood *= model.nbPosProbs[word]
      
        negLikelihood *= model.nbNegProbs[word]

    prediction = '1' if posLikelihood > negLikelihood else '0'
    model.predictions[index] = prediction #store prediction for bootstrapping
    index += 1

# parameters: 
  #model = NBCount or NBBinary model
  #testFile = list representation of the test file
def fMeasureBootStrap(model, testFile): #method for f-measure that uses stored predictions for faster runtime
  test = testFile
  
  countTruePos = 0
  countFalsePos = 0
  countTrueNeg = 0
  countFalseNeg = 0
  
  for line in test:
    prediction = model.predictions[int(line)]
   
    if (prediction == model.truth[int(line)]):
      if(prediction == '1'): #ie. if (prediction == 1 and truth == 1)
        countTruePos += 1
      else: #ie. if (prediction == 0 and truth == 0)
        countTrueNeg += 1
    else: #ie. if (prediction != truth)
      if(prediction == '1'): #ie. if (prediction == 1 and truth == 0)
        countFalsePos += 1
      else: #ie. if (prediction == 0 and truth == 1)
        countFalseNeg += 1
        
  precision = countTruePos / (countTruePos + countFalsePos)
  recall = countTruePos / (countTruePos + countFalseNeg)
    
  f1Measure = (2 * precision * recall) / (precision + recall)
  return f1Measure

def bootStrap(model1, model2): #does bootstrap sampling for 2 models
  test = []
  for i in range(400): #test set with just line number for faster runtime
    test.append(i)
  model1FMeasure = fMeasureBootStrap(model1, test) 
  model2FMeasure = fMeasureBootStrap(model2, test)
  deltaFMeasure = model1FMeasure - model2FMeasure #get the difference between the two files
  flipped = False
  if(deltaFMeasure < 0): #get absolute value if model 1 is worse than model 2, so we never get negative delta F measures, keep track if we flip the model
    flipped = True
    deltaFMeasure = abs(deltaFMeasure)
  
  deltaBSum = 0 

  b = 1000 #b = 1000 for bootstrap test
  for i in range (b): #create 1000 new test sets for the comparison
    newTestSet = []
    for i in range(400):
      temp = random.randrange(400) #0-399
      newTestSet.append(test[temp]) #add random line to new test set
    model1FMeasureB = fMeasureBootStrap(model1, newTestSet)
    model2FMeasureB = fMeasureBootStrap(model2, newTestSet)
    if(flipped):
      deltaFMeasureB = model2FMeasureB - model1FMeasureB
    else:
      deltaFMeasureB = model1FMeasureB - model2FMeasureB
      
    #if d(x(i) >= 2(d(x)) then add 1 to sum
    if(deltaFMeasureB >= (2*deltaFMeasure)):
      deltaBSum += 1
  pValue = deltaBSum / b #calculate the p-value for the model comparison

  return deltaFMeasure, pValue, flipped #return deltaF, pValue, and whether or not the comparison got flipped

#create plot1 and plot2
  #parameter file is name of the output file represented in a list of lines
def plot(file):
   X = [] #delta-fmeasure values
   Y = [] #1 - pvalue
   for line in file: 
     splitLine = line.split()
     X.append(float(splitLine[6]))  
     Y.append(1 - float(splitLine[8]))

   plt.xlabel("delta F1")
   plt.ylabel("1 - p-value")
   plt.scatter(X,Y, marker="x", c="black")
   plt.title("Plot 1")
   plt.savefig("plot1.pdf")

   Xsame = [] #delta-fmeasure values
   Ysame = [] #1 - pvalue
   Xdiff = []
   Ydiff = []
   for line in file:
     splitLine = line.split()
     if (splitLine[0] == splitLine[3]): #same type of model
       Xsame.append(float(splitLine[6]))
       Ysame.append(1 - float(splitLine[8]))
     else:
       Xdiff.append(float(splitLine[6]))
       Ydiff.append(1 - float(splitLine[8]))

   plt.clf()
   plt.xlabel("delta F1")
   plt.ylabel("1 - p-value")
   plt.scatter(Xdiff,Ydiff, facecolors='none', edgecolors='r')
   plt.scatter(Xsame,Ysame, marker="x", c="blue", edgecolors='b')

  
   plt.title("Plot 2")
   plt.legend(["red O -> different data representations", "blue X -> same data representations"], bbox_to_anchor=(1,1), loc="upper left")
   plt.savefig("plot2test.pdf", bbox_inches='tight')

    
def main(trainFiles, testFile):
  nbCountModels = []
  nbBinaryModels = []

  #creates all NBCount and NBBinary models, 10 for each different size
  for i in range (10):
    trainFile = open(str(trainFiles) + "/size650TrainingSets/train" + str(i + 1) + ".txt", 'r').readlines()
    nbC = NBCount(trainFile)
    nbB = NBBinary(trainFile)
    nbCountModels.append(nbC)
    nbBinaryModels.append(nbB)


  for i in range (10):
    trainFile = open(str(trainFiles) + "/size1300TrainingSets/train" + str(i + 1) + ".txt", 'r').readlines()
    nbC = NBCount(trainFile)
    nbB = NBBinary(trainFile)
    nbCountModels.append(nbC)
    nbBinaryModels.append(nbB)

  for i in range (10):
    trainFile = open(str(trainFiles) + "/size2600TrainingSets/train" + str(i + 1) + ".txt", 'r').readlines()
    nbC = NBCount(trainFile)
    nbB = NBBinary(trainFile)
    nbCountModels.append(nbC)
    nbBinaryModels.append(nbB)

  test = open(str(testFile), 'r').readlines()
    
  for i in range(30): #initalize the truths and predictions, so we dont have to     recalculate wheen bootstrapping (original f-measure method does this)
    populateProbs(nbCountModels[i], test)
    populateProbs(nbBinaryModels[i], test)

  D3 = open("./D3Output.txt", "wt") #where the results will be stored from the 1770 pairs
  
  #loop that compares all models, totaling 1770 comparisons. Flip the models to avoid negative delta F measures
  for i in range(30):
    for j in range(30):
      if j > i:
        deltaF, p, flipped = bootStrap(nbCountModels[i], nbCountModels[j])
        if(flipped):
          line = "NBCountModel " + str(j) + " vs. NBCountModel " + str(i) + ", deltaF: " + str(deltaF) + " p-value: " + str(p) + "\n"
          D3.write(line)
        else:
          line = "NBCountModel " + str(i) + " vs. NBCountModel " + str(j) + ", deltaF: " + str(deltaF) + " p-value: " + str(p) + "\n"
          D3.write(line)
        
        deltaF, p, flipped = bootStrap(nbBinaryModels[i], nbBinaryModels[j])
        if(flipped):
          line = "NBBinaryModel " + str(j) + " vs. NBBinaryModel " + str(i) + ", deltaF: " + str(deltaF) + " p-value: " + str(p) + "\n"
          D3.write(line)
        else:
          line = "NBBinaryModel " + str(i) + " vs. NBBinaryModel " + str(j) + ", deltaF: " + str(deltaF) + " p-value: " + str(p) + "\n"
          D3.write(line)
        
      deltaF, p, flipped = bootStrap(nbCountModels[i], nbBinaryModels[j])
      if(flipped):
      	line = "NBBinaryModel " + str(j) + " vs. NBCountModel " + str(i) + ", deltaF: " + str(deltaF) + " p-value: " + str(p) + "\n"
      	D3.write(line)
      else:
      	line = "NBCountModel " + str(i) + " vs. NBBinaryModel " + str(j) + ", deltaF: " + str(deltaF) + " p-value: " + str(p) + "\n"
      	D3.write(line)

      print(str(i) + " vs. " + str(j))

  D3.close()
  
  #can call the plotting function using this line, passing in the name of the outputfile
  #plot(open("./D3Output.txt").readlines())
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_files", help="Enter a path to the directory containing training files.") 
  parser.add_argument("--test_path", help="Enter a path to the test file.")
  args = parser.parse_args()
  main(args.train_files, args.test_path)

