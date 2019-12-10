# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util
import time

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()
  # print("datum.getPixels: " + str(a))
  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0

  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  Diagonal Feature Extraction:
  1. Split image into zones
  2. for each zone sum along diagonals and average all totals to get one feature


  ref: https://pdfs.semanticscholar.org/ddd8/29ecd1c1fdef2b04d2b61027a68e316e170e.pdf
  'Diagonal Feature Extraction Based Handwritten Character System Using Neural Network'


  Also check for loops as only some numbers have loops. 
  ##
  """

  a = datum.getPixels()
  features = util.Counter()

  closedLoop = False

  N = 4    # how many times to replicate non zero features

  i = 0
  for x in range(0,DIGIT_DATUM_WIDTH, 2):
    for y in range(0, DIGIT_DATUM_HEIGHT, 2):
      d1 = datum.getPixel(x,y)
      d2 = datum.getPixel(x+1, y) + datum.getPixel(x, y+1)
      # d2 = d2 /2
      d3 = datum.getPixel(x+1, y+1)
      avg = d1+d2+d3 / 3
      # print("(x,y) = " + str((x,y)) + " avg = " + str(avg))
      features[i] = avg
      if avg > 0:
        for n in range(1,N):
          features[i+n] = avg
      i += N
  
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      visited={}
      if datum.getPixel(x,y) == 0:
        if dfs(datum, x,y,visited):
          closedLoop=True
          break
  if closedLoop:
    for n in range(1,N):
      features["containsLoop"+str(n)] = 1

  vertScore = calcSymmetryScoreVert(datum)
  horiScore = calcSymmetryScoreHori(datum)

  # print("H Score: " + str(horiScore))
  # print("V Score: " + str(vertScore))
  features['vertical'] = vertScore
  features['horizontal'] = horiScore


  if vertScore > 0.5:
    for n in range(1,N):
      features['vertical'+str(n)] = vertScore
  if horiScore > 0.5:
    for n in range(1,N):
      features['horizontal'+str(n)] = horiScore

  horizontalEdgeProportion = lookForHorizontalEdges(datum, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
  
  for n in range(1,N):
    features['edge'+str(n)] = horizontalEdgeProportion

  verticalEdgeProportion = lookForVerticalEdges(datum, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
  
  for n in range(1,N):
    features['edge'+str(n)] = verticalEdgeProportion

  return features

def calcSymmetryScoreVert(datum):
  score = 0
  count = 0
  for x in range(int(DIGIT_DATUM_WIDTH/2)):
    for y in range(DIGIT_DATUM_HEIGHT):
      # print("x,y " + str(x) + ", " + str(y))
      p = datum.getPixel(x,y)
      q = datum.getPixel(DIGIT_DATUM_WIDTH-x-1,y)
      count+=1
      if p == q:
        score += 1

  return score/count

def calcSymmetryScoreHori(datum):
  score = 0
  count = 0
  for y in range(0,int(DIGIT_DATUM_HEIGHT/2)):
    for x in range(DIGIT_DATUM_WIDTH):
      p = datum.getPixel(x,y)
      q = datum.getPixel(x,DIGIT_DATUM_HEIGHT-y-1)
      count += 1
      if p == q:
        score += 1

  return score/count


def dfs(datum, x,y, visited):
  if x<0 or x>=DIGIT_DATUM_WIDTH:
    return False
  if y<0 or y>=DIGIT_DATUM_HEIGHT:
    return False
  if datum.getPixel(x,y) > 0:
    return True
  if (x,y) in visited.keys():
    return True
  visited[(x,y)] = True
  return dfs(datum, x-1,y, visited) and dfs(datum, x+1,y,visited) and dfs(datum, x, y-1,visited) and dfs(datum, x, y+1,visited)

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  a = datum.getPixels()
  features = util.Counter()
  closedLoop = False

  #dim of each zone ... should be a factor of 28
  m = 2
  N = 10 # how many times to replicate non zero features

  i = 0
  for x in range(0,DIGIT_DATUM_WIDTH, m):
    for y in range(0, DIGIT_DATUM_HEIGHT, m):
      d1 = datum.getPixel(x,y)
      d2 = datum.getPixel(x+1, y) + datum.getPixel(x, y+1)
      # d2 = d2 /2
      d3 = datum.getPixel(x+1, y+1)
      avg = d1+d2+d3 / 3
      # print("(x,y) = " + str((x,y)) + " avg = " + str(avg))
      features[i] = avg
      if avg > 0:
        for n in range(1,N):
          features[i+n] = avg
      i += N

  vertScore = calcSymmetryScoreVert(datum)
  horiScore = calcSymmetryScoreHori(datum)

  # print("H Score: " + str(horiScore))
  # print("V Score: " + str(vertScore))
  features['vertical'] = vertScore
  features['horizontal'] = horiScore


  if vertScore > 0.5:
    for n in range(1,N):
      features['vertical'+str(n)] = vertScore
  if horiScore > 0.5:
    for n in range(1,N):
      features['horizontal'+str(n)] = horiScore

  horizontalEdgeProportion = lookForHorizontalEdges(datum, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
  features['edge'] = horizontalEdgeProportion
  if horizontalEdgeProportion > 0.5:
    for n in range(1,N):
      features['edge'+str(n)] = horizontalEdgeProportion

  verticalEdgeProportion = lookForVerticalEdges(datum, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
  features['edge'] = verticalEdgeProportion
  if verticalEdgeProportion > 0.5:
    for n in range(1,N):
      features['edge'+str(n)] = verticalEdgeProportion

  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      visited={}
      if datum.getPixel(x,y) == 0:
        if dfs(datum, x,y,visited):
          closedLoop=True
          break
  if closedLoop:
    for n in range(1,N):
      features["containsLoop"+str(n)] = 1
  return features

def lookForHorizontalEdges(datum, width, height):
  score = 0
  count = 0
  maxEdgeProportion = -1
  for x in range(width):
    for y in range(height):
      p = datum.getPixel(x,y)
      countToTheLeft = lookLeft(datum, x, y)
      countToTheRight = lookRight(datum, x, y, width)
      edgeProportion = float(countToTheRight + countToTheLeft)/width
      if edgeProportion > maxEdgeProportion:
        maxEdgeProportion = edgeProportion

  return maxEdgeProportion

def lookLeft(datum, x, y):
  p = datum.getPixel(x,y)
  count = 0
  i = x
  while i>0 and datum.getPixel(i,y) == p:
    count += 1
    i += -1

  return count

def lookRight(datum, x, y, width):
  p = datum.getPixel(x,y)
  count = 0
  i = x
  while i<width and datum.getPixel(i,y) == p:
    count += 1
    i += 1

  return count
def lookForVerticalEdges(datum, width, height):
  score = 0
  count = 0
  maxEdgeProportion = -1
  for x in range(width):
    for y in range(height):
      p = datum.getPixel(x,y)
      countUp = lookUp(datum, x, y)
      countDown = lookDown(datum, x, y, height)
      edgeProportion = float(countUp + countDown)/height
      if edgeProportion > maxEdgeProportion:
        maxEdgeProportion = edgeProportion

  return maxEdgeProportion

def lookUp(datum, x, y):
  p = datum.getPixel(x,y)
  count = 0
  i = x
  while i>0 and datum.getPixel(x,i) == p:
    count += 1
    i += -1

  return count

def lookDown(datum, x, y, height):
  p = datum.getPixel(x,y)
  count = 0
  i = x
  while i<height and datum.getPixel(x,i) == p:
    count += 1
    i += 1

  return count
def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print( "===================================")
          print( "Mistake on example %d" % i )
          print ("Predicted %d; truth is %d" % (prediction, truth))
          print ("Image: ")
          print (rawTestData[i])
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print( image)  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print("Doing classification")
  print ("--------------------")
  print ("data:\t\t" + options.data)
  print ("classifier:\t\t" + options.classifier)
  if not options.classifier == 'minicontest':
    print ("using enhanced features?:\t" + str(options.features))
  else:
    print ("using minicontest feature extractor")
  print ("training set size:\t" + str(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print ("Unknown dataset", options.data)
    print (USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print ("Training set size should be a positive integer (you provided: %d)" % options.training)
    print (USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print ("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print (USAGE_STRING)
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print ("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
      print (USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "mostFrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print ("using automatic tuning for naivebayes")
        classifier.automaticTuning = True
    else:
        print ("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "mira"):
    classifier = mira.MiraClassifier(legalLabels, options.iterations)
    if (options.autotune):
        print("using automatic tuning for MIRA")
        classifier.automaticTuning = True
    else:
        print ("using default C=0.001 for MIRA")
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print ("Unknown classifier:", options.classifier)
    print (USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    
  

  ###
  ##Start Timing
  ###
  start_time = time.time()
  # Extract features
  print ("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))


  # print("Test Labels: " + str(testLabels))
  # Conduct training and testing
  print ("Training...")
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)

  ###
  ##Start Timing
  ###
  end_time = time.time()

  print ("Validating...")
  guesses = classifier.classify(validationData)
  # print("Guesses: " + str(guesses))
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
  print ("Testing...")

  classification_time_start = time.time()


  guesses = classifier.classify(testData)

  classification_time_end = time.time()

  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print (str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
  # analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  print("Time To Train: " + str(end_time - start_time))
  print("Time To Classify: " + str(classification_time_end - classification_time_start))


  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)
