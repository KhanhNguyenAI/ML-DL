import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import tensorflow as tf  
from keras.models import load_model
url = r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\diabetes\csv\diabetes.csv'
df = pd.read_csv(url)
# print(df.head())
# print(df.info()) 
# print(df.describe())
# print(df.columns)#[8 rows x 9 columns]
# print(df.shape) #(768, 9)

#===missing data ====#
# print(df.isnull().sum().sum())
# print(df.isna().sum())
#=== 
X = df.drop(columns='Outcome',axis=0)
y = df['Outcome']
#==========================
# for i in range(len(df.columns[:-1])):
#     label = df.columns[i]
#     plt.hist(df[df['Outcome']==1][label], color='blue', label="Diabetes", alpha=0.7, density=True, bins=15)
#     plt.hist(df[df['Outcome']==0][label], color='red', label="No diabetes", alpha=0.7, density=True, bins=15)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()
# print(df[df['Outcome'] ==1].shape) #268
# print(df[df['Outcome'] ==0].shape)#500
#=============STD Scaler =========#
from sklearn.preprocessing import StandardScaler 
scaler_sdt = StandardScaler()
X = scaler_sdt.fit_transform(X)
#=========imbalance
from imblearn.over_sampling import RandomOverSampler
over = RandomOverSampler()
X, y = over.fit_resample(X, y)
# print(y[y==1].shape)
# print(y[y==0].shape)
#=================================split======
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)
# print(X_train.shape,X_test.shape) #537,116,115
# print(y_test.shape,y_test.shape)
#==================Kford============ because X_train too small ==> Xtrain + Xvail to train model

def model_1(): 
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    model1 = LogisticRegression(solver='liblinear',random_state=42)
    model1.fit(X_train,y_train)
    y_pre = model1.predict(X_test)
    print(classification_report(y_test,y_pre))


model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_dim =8  ),
          tf.keras.layers.Dense(16,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])
# print(model.evaluate(X_train,y_train))
# print('-'*10)
history = model.fit(X_train,y_train,epochs=20,batch_size = 16,validation_split =0.15)
# print(model.evaluate(X_test,y_test))
model.save('modeldiabetes.h5')