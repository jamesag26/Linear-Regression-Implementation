
import numpy as np
import matplotlib.pyplot as plt

def getkfolds(k,dataList):
    # Places all of training data in one list to be randomly ditributed into k sets
    # Randomly shuffles data
    np.random.shuffle(dataList)
    # Finds size of K equal sets
    lenPart=int(np.math.ceil(len(dataList)/float(k)))
    # Divides dataset into k partitions of size lenPart
    partTrain = {}
    for x in range(k):
        partTrain[x] = dataList[x*lenPart:x*lenPart+lenPart]
    return partTrain

def fit_lr_normal(data, labels):
    # Transpose of the data matrix
    dataT = np.transpose(data)
    # Finds the normal equation values for model
    theta1 = np.linalg.inv(dataT.dot(data)).dot(dataT).dot(labels)
    return theta1
    
def predict_lr(model, data):
    temp = 0
    # Predicts first element since I was getting errors for array for vstack when array started empty
    for theta in range(0,len(model)):
        temp = temp + (model[theta] * data.item(0,theta))
    predLabels = temp
    # Predicts rest of the values for the model and data
    for x in range(1,len(data)):
        temp = 0
        for theta in range(0,len(model)):
            temp = temp + (model[theta] * data.item(x,theta))
        predLabels = np.vstack((predLabels,temp))
    predLabels = np.matrix(predLabels)
    return predLabels
    
def compute_mse(labels_actual,labels_estimated):
    mseSum = 0
    # Sums the square of the difference between actual and estimated
    for x in range(0,len(labels_actual)):
        mseSum = mseSum + (labels_actual[x] - labels_estimated[x]) ** 2
    # Finds the mse by dividing by number of difference values summed
    mse = mseSum / (len(labels_actual))
    return mse

# Test and train data contains 2 columns 
houseData = np.loadtxt('housing.data', dtype='float')
labelsTotal = [houseData[0][13]]
for y in range(1,len(houseData)):
    labelsTotal = np.vstack((labelsTotal,houseData[y][13]))

# get 6 folds of data
partData = getkfolds(6,houseData)
trainData = np.vstack((partData[0],partData[1]))
bestMSE = 100
bestModel = [0,0,0]
bestX = -1
bestY = -1
# First feature
for x in range(0,13):
    # Second feature
    for y in range(x+1,13):
        # For each k-fold 
        for k in range(0,6):
            # Test data
            testData = np.vstack((partData[k]))
            # Train data
            if (k == 0):
                trainData = np.vstack((partData[1],partData[2],partData[3],partData[4],partData[5]))
            elif (k ==1):
                trainData = np.vstack((partData[0],partData[2],partData[3],partData[4],partData[5]))
            elif (k ==1):
                trainData = np.vstack((partData[0],partData[1],partData[3],partData[4],partData[5]))
            elif (k ==1):
                trainData = np.vstack((partData[0],partData[1],partData[2],partData[4],partData[5]))
            elif (k ==1):
                trainData = np.vstack((partData[0],partData[1],partData[2],partData[3],partData[5]))
            else:
                trainData = np.vstack((partData[0],partData[1],partData[2],partData[3],partData[4]))
            # Get desired features and labels for trainData and testData
            trainDataFeats = [1,trainData[0][x],trainData[0][y]]
            trainLabels = [trainData[0][13]]
            testDataFeats = [1,testData[0][x],testData[0][y]]
            testLabels = [testData[0][13]]
            for z in range(1,len(trainData)):
                temp = [1,trainData[z][x],trainData[z][y]]
                trainDataFeats = np.vstack((trainDataFeats,temp))
                trainLabels = np.vstack((trainLabels,trainData[z][13]))
            for z in range(1,len(testData)):
                temp = [1,testData[z][x],testData[z][y]]
                testDataFeats = np.vstack((testDataFeats,temp))
                testLabels = np.vstack((testLabels,testData[z][13]))
            model = fit_lr_normal(trainDataFeats, trainLabels)
            predLabels = predict_lr(model,testDataFeats)
            mse = compute_mse(testLabels,predLabels)
            print("MSE for features ",x," and ",y, " is ", mse.item(0))
            if (mse < bestMSE):
                bestMSE = mse
                bestModel = model
                bestX = x
                bestY = y
                
                
print("The best MSE is: ",bestMSE)
print("The best model is: ", bestModel)                    
print("The two best features are ",bestX, " and ", bestY) 

# Test over entire dataset
totalTestData = [1,houseData[0][bestX],houseData[0][bestY]]
for z in range(1,len(houseData)):
    temp = [1,houseData[z][bestX],houseData[z][bestY]]
    totalTestData = np.vstack((totalTestData,temp))
predLabelsTotal = predict_lr(bestModel,totalTestData)
mseTotal = compute_mse(labelsTotal,predLabelsTotal)   
print("The MSE for the best model over all data is ",mseTotal)            
            
#print(houseData)
#np.random.shuffle(houseData)
#print(houseData)