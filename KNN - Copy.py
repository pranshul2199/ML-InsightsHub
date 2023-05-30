


import pandas as pd
df = pd.read_csv('heart.csv')


print("Without any reduction technique: -------")
print('Algorithm = KNN\n' )
#%% dividing target and features
X= df.iloc[:,:13]
y= df.iloc[:,13]




#%% machine learning
from sklearn.model_selection import train_test_split

Y=y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#%% algo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

# Create Decision Tree classifer object
clf = KNeighborsClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
a=cm[0][0]
b=cm[0][1]
c=cm[1][0]
d=cm[1][1]
Err=(b+c)/(a+b+c+d)
Acc=(a+d)/(a+b+c+d)
SN=a/(a+c)
Prec=a/(a+b)
print(cm)
print('Accuracy=',Acc*100,'%')
print('Error=',Err*100,'%')
print('Sensitivity=',SN*100,'%')
print('Prediction=',Prec*100,'%')

#%%
print("\n\nReduction Technique : PCA")
print('Algorithm = KNN\n' )
#%% dividing target and features
X= df.iloc[:,:13]
y= df.iloc[:,13]


#%% Feature engineering (Apply PCA algo)
from sklearn.decomposition import PCA


pca = PCA(n_components=1)


pca_X=pca.fit_transform(X,y)


X=pca_X

#%% machine learning
from sklearn.model_selection import train_test_split

Y=y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#%% algo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

# Create Decision Tree classifer object
clf = KNeighborsClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
a=cm[0][0]
b=cm[0][1]
c=cm[1][0]
d=cm[1][1]
Err=(b+c)/(a+b+c+d)
Acc=(a+d)/(a+b+c+d)
SN=a/(a+c)
Prec=a/(a+b)
print(cm)
print('Accuracy=',Acc*100,'%')
print('Error=',Err*100,'%')
print('Sensitivity=',SN*100,'%')
print('Prediction=',Prec*100,'%')


#%%
print("\n\nReduction Technique: LDA")
print('Algorithm = KNN\n' )
#%% dividing target and features
X= df.iloc[:,:13]
y= df.iloc[:,13]


#%% LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# # define transform
lda = LinearDiscriminantAnalysis(n_components=1)
# prepare transform on dataset
lda_X =lda.fit_transform(X,y)

X=lda_X

#%% machine learning
from sklearn.model_selection import train_test_split

Y=y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#%% algo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier 

# Create Decision Tree classifer object
clf = KNeighborsClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
a=cm[0][0]
b=cm[0][1]
c=cm[1][0]
d=cm[1][1]
Err=(b+c)/(a+b+c+d)
Acc=(a+d)/(a+b+c+d)
SN=a/(a+c)
Prec=a/(a+b)
print(cm)
print('Accuracy=',Acc*100,'%')
print('Error=',Err*100,'%')
print('Sensitivity=',SN*100,'%')
print('Prediction=',Prec*100,'%')

#%%
print("n\nReduction Technique: MLE")
print('Algorithm = KNN\n' )
#%% dividing target and features
X= df.iloc[:,:13]
y= df.iloc[:,13]


#%%
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


mean= X.mean()
std=X.std()
normal=norm.cdf(X,loc=mean,scale=std)
df=pd.DataFrame(normal)
lh=np.product(df)
llh=np.log(lh)
median=llh.median()
llh=abs(llh)
median=abs(median)
c=llh.count()
n=int(c) 
a=[]
b=[]
k=0
for i in range(0, n):
    a.append(i)
if c>10:
    ax=5
else:
    ax=1 
'''plt.plot(a,lh,marker='o')
plt.xticks(np.arange(0,c+ax,ax))
plt.show()'''
plt.plot(a,llh, marker='o')
plt.xticks(np.arange(0, c+ax,ax))
plt.show()
for j in range(0, n):
    if llh[j]>median:
        m=a[j]
        b.insert(k,m)
        k=k+1
print(k, " features/attributes selected")
print(b, "indices are selected")
X=df.iloc[:,b]

#%% machine learning
from sklearn.model_selection import train_test_split

Y=y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#%% algo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

# Create Decision Tree classifer object
clf = KNeighborsClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
a=cm[0][0]
b=cm[0][1]
c=cm[1][0]
d=cm[1][1]
Err=(b+c)/(a+b+c+d)
Acc=(a+d)/(a+b+c+d)
SN=a/(a+c)
Prec=a/(a+b)
print(cm)
print('Accuracy=',Acc*100,'%')
print('Error=',Err*100,'%')
print('Sensitivity=',SN*100,'%')
print('Prediction=',Prec*100,'%')
