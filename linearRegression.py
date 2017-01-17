
import numpy as np
import matplotlib.pyplot as plt

def fit_lr_normal(data, labels):
    # Transpose of the data matrix
    dataT = np.transpose(data)
    # Finds the normal equation values for model
    theta1 = np.linalg.inv(dataT * data) * dataT * labels
    return theta1
    
def fit_lr_gd(data, labels):
    alpha = .01
    m = len(labels)
    print (data.shape[1])
    # Order 0
    if data.shape[1] == 1:
        theta0 = 5
        tempModel = [[0]]
        model = np.matrix(tempModel)
        j_new = 10
        j_old = 0
        while abs(j_new - j_old) > .00001:
            predData = predict_lr(model,data)
            theta0 = theta0 - alpha * (1 / m) * sum(predData - labels)
            tempModel = theta0
            model = np.matrix(theta0)
            j_old = j_new
            tempPred = predict_lr(model,data)
            tempJ = compute_mse(labels, tempPred) / 2
            j_new = tempJ   
        return model
    # Order 1
    if data.shape[1] == 2:
        theta0 = 5
        theta1 = 5
        tempModel = [[0], [0]]
        model = np.matrix(tempModel)
        j_new = 10
        j_old = 0
        while abs(j_new - j_old) > .00001:
            predData = predict_lr(model,data)
            theta0 = theta0 - alpha * (1 / m) * sum(predData - labels)
            theta1Sum = 0
            for x in range(0,m):
                theta1Sum += (predData[x] - labels[x]) * data.item(x,1)
            theta1 = theta1 - alpha * (1 / m) * theta1Sum 
            tempModel = np.vstack((theta0,theta1))
            model = np.matrix(tempModel)
            j_old = j_new
            tempPred = predict_lr(model,data)
            tempJ = compute_mse(labels, tempPred) / 2
            j_new = tempJ 
        return model
    # Order 2
    if data.shape[1] == 3:
        theta0 = 5
        theta1 = 5
        theta2 = 5
        tempModel = [[0], [0], [0]]
        model = np.matrix(tempModel)
        j_new = 10
        j_old = 0
        while abs(j_new - j_old) > .00001:
            predData = predict_lr(model,data)
            theta0 = theta0 - alpha * (1 / m) * sum(predData - labels)
            theta1Sum = 0
            for x in range(0,m):
                theta1Sum += (predData[x] - labels[x]) * data.item(x,1)
            theta1 = theta1 - alpha * (1 / m) * theta1Sum 
            theta2Sum = 0
            for x in range(0,m):
                theta2Sum += (predData[x] - labels[x]) * data.item(x,2)
            theta2 = theta2 - alpha * (3 / (2 * m)) * theta2Sum 
            tempModel = np.vstack((theta0,theta1,theta2))
            model = np.matrix(tempModel)
            j_old = j_new
            tempPred = predict_lr(model,data)
            tempJ = compute_mse(labels, tempPred) / 2
            j_new = tempJ 
        return model       
    # Order 3
    if data.shape[1] == 4:
        theta0 = 5
        theta1 = 5
        theta2 = 5
        theta3 = 5
        tempModel = [[0], [0], [0], [0]]
        model = np.matrix(tempModel)
        j_new = 10
        j_old = 0
        while abs(j_new - j_old) > .00001:
            predData = predict_lr(model,data)
            theta0 = theta0 - alpha * (1 / m) * sum(predData - labels)
            theta1Sum = 0
            for x in range(0,m):
                theta1Sum += (predData[x] - labels[x]) * data.item(x,1)
            theta1 = theta1 - alpha * (1 / m) * theta1Sum 
            theta2Sum = 0
            for x in range(0,m):
                theta2Sum += (predData[x] - labels[x]) * data.item(x,2)
            theta2 = theta2 - alpha * (3 / (2 * m)) * theta2Sum 
            theta3Sum = 0
            for x in range(0,m):
                theta3Sum += (predData[x] - labels[x]) * data.item(x,3)
            theta3 = theta3 - alpha * (2 / m) * theta3Sum 
            tempModel = np.vstack((theta0,theta1,theta2,theta3))
            model = np.matrix(tempModel)
            j_old = j_new
            tempPred = predict_lr(model,data)
            tempJ = compute_mse(labels, tempPred) / 2
            j_new = tempJ 
        return model    
    # Order 4
    if data.shape[1] == 5:
        theta0 = 5
        theta1 = 5
        theta2 = 5
        theta3 = 5
        theta4 = 5
        tempModel = [[0], [0], [0], [0], [0]]
        model = np.matrix(tempModel)
        j_new = 10
        j_old = 0
        while abs(j_new - j_old) > .00001:
            predData = predict_lr(model,data)
            theta0 = theta0 - alpha * (1 / m) * sum(predData - labels)
            theta1Sum = 0
            for x in range(0,m):
                theta1Sum += (predData[x] - labels[x]) * data.item(x,1)
            theta1 = theta1 - alpha * (1 / m) * theta1Sum 
            theta2Sum = 0
            for x in range(0,m):
                theta2Sum += (predData[x] - labels[x]) * data.item(x,2)
            theta2 = theta2 - alpha * (3 / (2 * m)) * theta2Sum 
            theta3Sum = 0
            for x in range(0,m):
                theta3Sum += (predData[x] - labels[x]) * data.item(x,3)
            theta3 = theta3 - alpha * (2 / m) * theta3Sum 
            theta4Sum = 0
            for x in range(0,m):
                theta4Sum += (predData[x] - labels[x]) * data.item(x,4)
            theta4 = theta4 - alpha * (5 / (2 *m)) * theta4Sum 
            tempModel = np.vstack((theta0,theta1,theta2,theta3,theta4))
            model = np.matrix(tempModel)
            j_old = j_new
            tempPred = predict_lr(model,data)
            tempJ = compute_mse(labels, tempPred) / 2
            j_new = tempJ 
        return model   

    
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
trainData = np.loadtxt('hw2.train', dtype='float', delimiter=',')
testData = np.loadtxt('hw2.test', dtype='float', delimiter=',')

# Training data
# Predictor data
predVals = [trainData[0][0]]
# Target value
targVals = [trainData[0][1]]
for x in range (1,len(trainData)):
    predVals = np.vstack((predVals,trainData[x][0]))
    targVals = np.vstack((targVals,trainData[x][1]))

# Basis expansion order 0-4
temp0 = [1]
temp1 = [1,predVals[0] ** 1]
temp2 = [1,predVals[0] ** 1,predVals[0] ** 2]
temp3 = [1,predVals[0] ** 1,predVals[0] ** 2,predVals[0] ** 3]
temp4 = [1,predVals[0] ** 1,predVals[0] ** 2,predVals[0] ** 3,predVals[0] ** 4]
 
for x in range (1,len(predVals)):
    temp = [1]
    temp0 = np.vstack((temp0, temp))
    #temp = [1,predVals[x] ** 1]
    temp.append(predVals[x] ** 1)
    temp1 = np.vstack((temp1, temp))
    #temp = [1,predVals[x] ** 2]
    temp.append(predVals[x] ** 2)
    temp2 = np.vstack((temp2, temp))
    temp.append(predVals[x] ** 3)
    #temp = [1,predVals[x] ** 3]
    temp3 = np.vstack((temp3, temp))
    temp.append(predVals[x] ** 4)
    #temp = [1,predVals[x] ** 4]
    temp4 = np.vstack((temp4, temp))
# Setting up matrices    
mat0 = np.matrix(temp0)
mat1 = np.matrix(temp1)
mat2 = np.matrix(temp2)
mat3 = np.matrix(temp3)
mat4 = np.matrix(temp4)


# Test data
testPredVals = [testData[0][0]]
# Target value
testTargVals = [testData[0][1]]
for x in range (1,len(testData)):
    testPredVals = np.vstack((testPredVals,testData[x][0]))
    testTargVals = np.vstack((testTargVals,testData[x][1]))

# Basis expansion order 0-4
testTemp0 = [1]
testTemp1 = [1,testPredVals[0] ** 1]
testTemp2 = [1,testPredVals[0] ** 1,testPredVals[0] ** 2]
testTemp3 = [1,testPredVals[0] ** 1,testPredVals[0] ** 2,testPredVals[0] ** 3]
testTemp4 = [1,testPredVals[0] ** 1,testPredVals[0] ** 2,testPredVals[0] ** 3,testPredVals[0] ** 4]
 
for x in range (1,len(testPredVals)):
    testTemp = [1]
    testTemp0 = np.vstack((testTemp0, testTemp))
    testTemp.append(testPredVals[x] ** 1)
    testTemp1 = np.vstack((testTemp1, testTemp))
    testTemp.append(testPredVals[x] ** 2)
    testTemp2 = np.vstack((testTemp2, testTemp))
    testTemp.append(testPredVals[x] ** 3)
    testTemp3 = np.vstack((testTemp3, testTemp))
    testTemp.append(testPredVals[x] ** 4)
    testTemp4 = np.vstack((testTemp4, testTemp))
# Setting up matrices    
testMat0 = np.matrix(testTemp0)
testMat1 = np.matrix(testTemp1)
testMat2 = np.matrix(testTemp2)
testMat3 = np.matrix(testTemp3)
testMat4 = np.matrix(testTemp4)


model0n = fit_lr_normal(mat0, targVals)
predLabels0n = predict_lr(model0n,testMat0)
mse0n = compute_mse(testTargVals,predLabels0n)
print("MSE for 0-Order Normal Equation is: ", mse0n.item(0))

model1n = fit_lr_normal(mat1, targVals)
predLabels1n = predict_lr(model1n,testMat1)
mse1n = compute_mse(testTargVals,predLabels1n)
print("MSE for 1-Order Normal Equation is: ", mse1n.item(0))

model2n = fit_lr_normal(mat2, targVals)
predLabels2n = predict_lr(model2n,testMat2)
mse2n = compute_mse(testTargVals,predLabels2n)
print("MSE for 2-Order Normal Equation is: ", mse2n.item(0))

model3n = fit_lr_normal(mat3, targVals)
predLabels3n = predict_lr(model3n,testMat3)
mse3n = compute_mse(testTargVals,predLabels3n)
print("MSE for 3-Order Normal Equation is: ", mse3n.item(0))

model4n = fit_lr_normal(mat4, targVals)
predLabels4n = predict_lr(model4n,testMat4)
mse4n = compute_mse(testTargVals,predLabels4n)
print("MSE for 4-Order Normal Equation is: ", mse4n.item(0))

model0g = fit_lr_gd(mat0, targVals)
predLabels0g = predict_lr(model0g,testMat0)
mse0g = compute_mse(testTargVals,predLabels0g)
print("MSE for 0-Order Gradient Descent is: ", mse0g.item(0))

model1g = fit_lr_gd(mat1, targVals)
predLabels1g = predict_lr(model1g,testMat1)
mse1g = compute_mse(testTargVals,predLabels1g)
print("MSE for 1-Order Gradient Descent is: ", mse1g.item(0))

model2g = fit_lr_gd(mat2, targVals)
print(model2g)
predLabels2g = predict_lr(model2g,testMat2)
mse2g = compute_mse(testTargVals,predLabels2g)
print("MSE for 2-Order Gradient Descent is: ", mse2g.item(0))

model3g = fit_lr_gd(mat3, targVals)
print(model3g)
predLabels3g = predict_lr(model3g,testMat3)
mse3g = compute_mse(testTargVals,predLabels3g)
print("MSE for 3-Order Gradient Descent is: ", mse3g.item(0))

model4g = fit_lr_gd(mat4, targVals)
print(model4g)
predLabels4g = predict_lr(model4g,testMat4)
mse4g = compute_mse(testTargVals,predLabels4g)
print("MSE for 4-Order Gradient Descent is: ", mse4g.item(0))

plt.figure(1)
plt.title("0-Order Normal Equations")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model0n + xx * 0)
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(2)
plt.title("1-Order Normal Equations")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model1n[0] + model1n[1] * xx)
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(3)
plt.title("2-Order Normal Equations")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model2n[0] + model2n[1] * xx + model2n[2] * (xx ** 2))
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(4)
plt.title("3-Order Normal Equations")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model3n[0] + model3n[1] * xx + model3n[2] * (xx ** 2) + model3n[3] * (xx ** 3))
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(5)
plt.title("4-Order Normal Equations")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model4n[0] + model4n[1] * xx + model4n[2] * (xx ** 2) + model4n[3] * (xx ** 3) + model4n[4] * (xx ** 4))
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()
'''
'''
plt.figure(6)
plt.title("0-Order Gradient Descent")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model0g + xx * 0)
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(7)
plt.title("1-Order Gradient Descent")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model1g[0] + model1g[1] * xx)
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(8)
plt.title("2-Order Gradient Descent")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model2g[0] + model2g[1] * xx + model2g[2] * (xx ** 2))
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(9)
plt.title("3-Order Gradient Descent")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model3g[0] + model3g[1] * xx + model3g[2] * (xx ** 2) + model3g[3] * (xx ** 3))
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()

plt.figure(10)
plt.title("4-Order Gradient Descent")
xx = np.linspace(min(testPredVals), max(testPredVals))
yy = np.array(model4g[0] + model4g[1] * xx + model4g[2] * (xx ** 2) + model4g[3] * (xx ** 3) + model4g[4] * (xx ** 4))
plt.plot(xx, yy.T, color='b')
plt.scatter(predVals, targVals, c='green', alpha=0.5)
plt.scatter(testPredVals, testTargVals, c='red', alpha=0.5)
plt.show()