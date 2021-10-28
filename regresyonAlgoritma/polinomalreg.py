
#%% goruntuleme
import seaborn as sns

import pandas as pd

data=sns.load_dataset("tips")

df=pd.DataFrame(data)

print(df)

# eksik verileri doldurma!!

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(df[["total_bill"]])
df[["total_bill"]] = imputer.transform(df[["total_bill"]])

#impute  eksik verileri o sutun ort ile dolduruyor


#kategorik olmayan degiskenler--> sayısal veriler


total_bill = df[["total_bill"]]
tip = df[["tip"]]
size = df[["size"]]

#kategorik degiskenler


import pandas as pd

sm=pd.get_dummies(df[["smoker"]])

time=pd.get_dummies(df[["day"]])
results=df["day"].value_counts()
print(results)

#her bir günden kac tane var?????


print(sm)
#smoker 1 0  oılarak kukla degiskeni kaldırmalı


sm=sm.drop("smoker_Yes",axis=1)


# verileri birlestirme

new_data=pd.concat([total_bill,size,sm,time,tip],axis=1)
#tip *tahmin icin


#train ve test


tr=new_data.iloc[:,0:7]
#tip en son  tahmin edilmek istenen haric

test=new_data.iloc[:,7:8]


#train test split



from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test=train_test_split(tr,test,test_size=0.3,random_state=42)



from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()
linear_reg.fit(x_train,y_train)

#polinomal features

from sklearn.preprocessing import PolynomialFeatures

pol=PolynomialFeatures(degree=3) 
#3 dereceden pol

xp=pol.fit_transform(x_train)
#train


line=LinearRegression()
line.fit(xp,y_train)
#tahmin
print(line.predict(pol.fit_transform([[16.99,2,1,0,0,0,1]])))





