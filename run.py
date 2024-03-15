# import the required packages here
import numpy as np

# function for the actual predicting after the trianing 
def predict(weights, inputData):

  labels = []

  for i in range(len(weights)):
    labels.append(weights[i][1] * np.sign(np.dot(weights[i][0], inputData)))

  if np.sign(np.sum(labels)) == 1:
    return 1
  else:
    return 0

# def step(yHat):
#   if yHat > 0: #sign() == positive num 0 or 1
#     return 1 # positive classification 
#   else:
#     return 0  # negative classification 

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
  # Each row is a feature vector. The values in the -th columns are integer values in the -th dimension.
  trainX = np.loadtxt(Xtrain_file, delimiter = ",")
  
  # The CSV file provides the binary labels for corresponding feature vectors in the file
  trainY = np.loadtxt(Ytrain_file)

  # correct 0 lables in trainY 0 = -1
  for i in range(len(trainY)):
    if trainY[i] == 0:
      trainY[i] = -1

  # print test
  # print("np array trainX: \n", trainX, trainX.shape, "\n np array trainY: \n", trainY, trainY.shape)
  # print("\n", trainY[0])

  # t_i = y = labels = {-1, +1} = trainY but for this project {0, 1}
  # x_i = feature vectors = trainX input 
  # T = number of epochs aka number of times the algorithm iterates fully through the data

  # making weighted vector output array and output label array that starts as a 0 feature vector 
  xColNum = len(trainX[0]) # gave me size error when making w same col as trainX
  xRowNum = len(trainX)

  # c[0] = 0
  # w[0] = 0
  w = np.zeros((xRowNum, xColNum)) # making it as big as trainX[] row size for dot product to work
  c = np.zeros(len(trainY))
  # print("w:\n", w, w.shape)
  # print("c:\n", c, c.shape)
  # print("w[0]:\n", w[0])
  # print("c[0]:\n", c[0])

  # print test
  # print("np array w: \n", w, "with length: ", len(w), "\n np array c: \n", c, "with length: ", len(c))

  # w_k = weighted vector output
  # c_k = weighted vector's output label 

  t = 0 # count
  T = 20 # epochs

  # y_hat == prediction

  # print("test: ", np.dot(w[k], trainX[1]) * trainY[1])

  while t <= T: # t is count of total full iterations, T is max amount of iterations wanted aka epoch

    k = 0 # number of weight vectors w aka perceptrons 

    for i in range(len(trainX)):  # iterate thru entire data set starting from i = 0
      # wSum = (w, trainX)) # weighted sum
      # print(wSum)

      dot = np.dot(w[k], trainX[i]) # dot product of w[k] and x[i]
      # print("dot: ", dot)

      yHat = np.sign(dot) # setting the prediction yHat to be the sign of the dot
      # print("yHat: ", yHat)

      if yHat != trainY[i]:  # if not correctly classified 
        w[k + 1] = w[k] + (trainY[i] * trainX[i])
        #np.put(w, k+1, w[k] + (trainY[i] * trainX[i]))
        # np.put(c, i+1, 1) 
        c[k + 1] = 1
        k = k + 1 # k++
      else: # if correctly classified 
        # print(c[k])
        c[k] += 1 #c[k] ++
        #np.put(c, k, c[k] + 1)
      
      # if prediction == trainY[i] or t == T: 
      #   print("Correct")
      # else: 
      #   print("Not Correct")

    t = t + 1 #t++
    # print("t: ", t)


  # w and c print test
  # print("\nend test: \nw:\n", w, "\n\n c:\n", c)

  perceptrons = np.array([[w_i, c_i] for w_i, c_i in zip(w, c)], dtype = object)

  # load in the test data
  testData = np.loadtxt(test_data_file, delimiter = ",")
  prediction = [predict(perceptrons, data) for data in testData]

  # Saving your prediction to pred_file directory (Saving can't be changed)
  np.savetxt(pred_file, prediction, fmt = '%1d', delimiter = ",")

  print("\nw print test:\n", w)

# if statement to test code
# if __name__ == "__main__":
#     Xtrain_file = 'Xtrain.csv'
#     Ytrain_file = 'Ytrain.csv'
#     test_input_dir = 'Xtrain.csv'
#     pred_file = 'prediction'
#     run(Xtrain_file, Ytrain_file, test_input_dir, pred_file)

if __name__ == "__main__":
  Ytrain_file = 'Ytrain.csv'
  pred_file = 'prediction'

  last_ten = 'lastTenPercent.csv'
  first_ninety = 'firstNinetyPercent.csv'

  # ////////////////// First, use the last 10% of the training data as your test data ////////////////////////////
  Xtrain_file = 'Xtrain.csv'
  trainX = np.loadtxt(Xtrain_file, delimiter = ",")
  # print(trainX)

  xColNum = len(trainX[0]) # gave me size error when making w same col as trainX
  xRowNum = len(trainX)

  total = int(len(trainX))
  tenPercent = int(.10 * total)
  # print(tenPercent)

  tenPercentCount = total - tenPercent  # counter to start at 90 percent of index
  # print("last 10 counter: ", tenPercentCount)

  # lastTenPercent = np.zeros((tenPercent, xColNum))
  lastTenPercent = trainX
  # print("last 10 before: ", lastTenPercent)

  for i in range(0, tenPercentCount):
    lastTenPercent = np.delete(lastTenPercent, 0, 0)
    # print(i)
  # print("last 10 after: ", lastTenPercent)

  # Saving as new file
  np.savetxt(last_ten, lastTenPercent, fmt = '%1d', delimiter = ",")
  test_input_dir = last_ten
  

  # ////////////////////////////////  first 90 percent  /////////////////////////////////////////////////// 
  firstNinetyPercent = trainX
  # print("first 90 before: ", firstNinetyPercent)

  for i in range(tenPercent):
    firstNinetyPercent = np.delete(firstNinetyPercent, tenPercentCount, 0)
    # print(i)
  # print("first 90 after: ", firstNinetyPercent)

  # Saving as new file
  # np.savetxt(first_ninety, firstNinetyPercent, fmt = '%1d', delimiter = ",")
  # Xtrain_file = first_ninety

  # /////////////////////// 1 2 5 10 20 and 100 percents of the 90 percent //////////////////////
  percents = [1, 2, 5, 10, 20, 100]
  for p in percents:
    p_percent = '%iPercent.csv' % p

    # print("p test: ", p)

    totalNinety = int(len(firstNinetyPercent))
    pPercent = p / 100
    # print("pPercent: ", pPercent)

    # index counter
    percentOfNinety =  int(pPercent * totalNinety)
    # print("percentOfNinety: ",percentOfNinety)
    
    # loop counter
    pPercentCount = totalNinety - percentOfNinety
    # print("loop counter:", pPercentCount)

    pTrainingData = firstNinetyPercent
    for i in range(pPercentCount):
      pTrainingData = np.delete(pTrainingData, percentOfNinety, 0)

    # print("after getting percent: ", pTrainingData)

    # Saving percentage as new file
    np.savetxt(p_percent, pTrainingData, fmt = '%1d', delimiter = ",")
    Xtrain_file = p_percent

    # //////////////////////// Accuracy Test //////////////////////
    run(Xtrain_file, Ytrain_file, test_input_dir, pred_file)
    pred = np.loadtxt(pred_file, skiprows = 0)
    actual = np.loadtxt(Ytrain_file, delimiter = ",")

    tp = 0
    tn = 0

    fp = 0
    fn = 0

    for a, pre in zip(actual, pred): 
      if a == 1 and pre == 1: 
        tp += 1
      elif a == 1 and pre == 0: 
        fn += 1
      elif a == 0 and pre == 1: 
        fp += 1
      else: 
        tn += 1
        
    accuracy = round(100 * (tp + tn) / (tp + fp + tn + fn), 4)

    print("\n---- Dataset %i Percent ----" % p)
    print("Accuracy: %s" % accuracy)
    print("TP : %i ; FP : %i" % (tp, fp))
    print("TN : %i ; FN : %i" % (tn, fn))
    print("\n") 
