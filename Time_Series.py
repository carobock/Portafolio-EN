import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import csv
from scipy import interpolate

data = pd.read_csv('Data.csv', sep=',', header=None)

training_data = data[[4,11,12,13]].head(int(0.8*data.shape[0]+1))
testing_data = data[[4,11,12,13]].drop(training_data.index)

y_train = np.array(training_data[4])
x_train = np.array(training_data[[11,12]])
y_test = np.array(testing_data[4])
x_test = np.array(testing_data[[11,12]])




reg= LinearRegression().fit(x_train, y_train)
#print(reg.coef_)
#print(reg.intercept_)

y_pred= reg.predict(x_test)

RMSE=mean_squared_error(y_pred, y_test, squared=False)
MAPE = mean_absolute_percentage_error(y_test, y_pred)

#print("The mean absolute percentage error is " + str(round(MAPE*100, 3))+"%")
#print("The root mean squared prediction error is " + str(round(RMSE, 3)))

#residuals = y_train-y_pred

#plt.scatter(y_pred,residuals)
#plt.show()

#PERIODIC SIGNAL

y_pred = reg.predict(x_train)
residuals = y_train-y_pred
month = np.array(data[1][:588])
P=[]
for i in range(1,13):
    mean_month_i = np.array([np.mean(residuals[np.where(month==i)[0]],0)])
    P.append(round(mean_month_i[0],3))

#print(P)

month = [1,2,3,4,5,6,7,8,9,10,11,12]
f = interpolate.interp1d(month,P)
ynew = f(month)
#plt.plot(month, P, 'o', month, ynew, '-')
#plt.xlabel("Months")
#plt.ylabel("Periodic Signal")
#for i,j in zip(month,P):
#    plt.annotate("  "+str(j),xy=(i,j))

#plt.show()

#FINAL MODEL

model= reg.intercept_+np.sum(data[[11,12]]*reg.coef_,axis=1)+np.array(data[14])
#plt.plot(range(len(model[:588])),model[:588], color="red")
#plt.plot(range(588,588+len(model[588:])),model[588:], color="black")
#plt.title("Final Fit")
#plt.ylabel("F(ti)+Pi")
#plt.xlabel("i")
#plt.text(4,400,"Training Data", color="red")
#plt.text(4,390,"Testing Data", color="black")
#plt.show()

#print(model)


y_pred_final= reg.predict(x_test)+data[14][588:]

RMSE=mean_squared_error(y_pred_final, y_test, squared=False)
MAPE = mean_absolute_percentage_error(y_test, y_pred_final)

print("The mean absolute percentage error is " + str(round(MAPE*100, 3))+"%")
print("The root mean squared prediction error is " + str(round(RMSE, 3)))
