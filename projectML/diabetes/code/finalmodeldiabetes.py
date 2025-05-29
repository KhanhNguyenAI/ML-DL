import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import tensorflow as tf  
from keras.models import load_model
url = r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\diabetes\csv\diabetes.csv'
df = pd.read_csv(url)
X = df.drop(columns='Outcome',axis=0)
y = df['Outcome']
from sklearn.preprocessing import StandardScaler 
scaler_sdt = StandardScaler()
X = scaler_sdt.fit_transform(X)
from imblearn.over_sampling import RandomOverSampler
over = RandomOverSampler()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)


model = load_model(r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\diabetes\model\modeldiabetes.h5')
loss,acc=model.evaluate(X_test,y_test)
X_new = X_test[10]
y_new = y_test[10]
# print(X_new)                #1D
X_new = np.expand_dims(X_new,axis=0)           #2D
# print(X_new)

y_pre = model.predict(X_new)
# print(y_pre)
# print(y_new)
if y_pre <= 0.5 : 
    print('no')
else : 
    print('Yes')
