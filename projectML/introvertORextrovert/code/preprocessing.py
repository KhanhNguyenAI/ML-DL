import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE



df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\introvertORextrovert\csv\personality_dataset.csv')
#print(df.shape) # (8) 
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
'''--------------------------------------------------------------
print(df.isna().sum()) # full 
Time_spent_Alone             63
Stage_fear                   73
Social_event_attendance      62
Going_outside                66
Drained_after_socializing    52
Friends_circle_size          77
Post_frequency               65
-------------------------------------------------------------- '''
# Fill missing values with the mean of each column
numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_columns = ['Stage_fear', 'Drained_after_socializing']
target_column = 'Personality'
'''
plt.figure(figsize=(8,6))
for i,col in enumerate(numeric_columns):
    plt.subplot(3,2,i+1)
    sns.boxplot(data=df,x = target_column,y=col)
    plt.title(f'{col} of Personality')
plt.tight_layout()
plt.show()
'''
'''
plt.figure(figsize=(8,6))
sns.countplot(data=df,x =target_column)
plt.title('class distribution of Personality Types')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()
'''
'''
sns.pairplot(df[numeric_columns + [target_column]],hue=target_column,diag_kind='hist')
plt.suptitle('Pair Plot of Numeric Features by Personality',y=1.02)
plt.show()

'''
# print(y[1])
label_enc = LabelEncoder()
df['Personality']= label_enc.fit_transform(df['Personality'])
 
'''0 = Extrovert
1 = Introvert

print(y[y==0].shape) # 0 = Extrovert
print(y[y==1].shape) # 1 = Introvert
>>>>> balanced dataset
'''




def each(df):
    figure, axes = plt.subplots(2, 4, figsize=(20, 10)) 
    for i, column in enumerate(X.columns):
        sns.histplot(X[column], kde=True, ax=axes[i // 4, i % 4])
        axes[i // 4, i % 4].set_title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()
def vitual(df):
    for col in X.columns:
        sns.violinplot(x=y, y=X[col], inner='quartile')
        plt.title(f'{col} by Personality Type')
        plt.show()

'''print(X.shape) (2900, 7)'''
df = df.dropna(axis=0, how='any') 
'''print(X.isna().sum()) 
print(X.shape) (2477, 7)'''
df['Stage_fear'] = label_enc.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = label_enc.fit_transform(df['Drained_after_socializing'])


'''
# Xem ma trận tương quan
corr_matrix = df.corr()
# print(corr_matrix)
sns.heatmap(corr_matrix,annot=True,fmt= '.2f',color='blue')
sns.pairplot(df_scaled)
plt.show() 
'''
'''sns.pairplot(df)
plt.show()'''

X = df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
            'Post_frequency']]
y = df['Personality']

'''
0 : yes
1 : no 
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=0.4,random_state=0)

# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_val)
model_lo = LogisticRegression()
model_lo.fit(X_train,y_train)
y_pre = model_lo.predict(X_val)
print(classification_report(y_val,y_pre))
'''
print(type(X_test))
print(X_test.iloc[0][:])
'''

sample = 10
Ex = X_test.iloc[sample][:].values.reshape(1, -1)
print(Ex)
Ex = pd.DataFrame(Ex, columns=X_train.columns)
print(type(Ex))
print(Ex.shape)

print(model_lo.predict(Ex))
print('='*30)
print(y_test.iloc[sample])      

# '''
# import pickle
# path = r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\introvertORextrovert\model"'
# with open("logistic_model.pkl", "wb") as f:
# pickle.dump(model_lo, f)
# '''
