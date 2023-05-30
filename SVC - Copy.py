


import pandas as pd
df = pd.read_csv('heart.csv')


print("Without any reduction technique: -------")
print('Algorithm = SVC\n' )
#%% dividing target and features
X= df.iloc[:,:13]
y= df.iloc[:,13]




#%% machine learning
from sklearn.model_selection import train_test_split

Y=y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#%% algo

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(X_train, Y_train)
# score = svc_classifier.score(X_test, Y_test)


y_pred = svc.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
c = confusion_matrix(Y_test, y_pred)
TN = c[0][0]
FP = c[0][1]
FN = c[1][0]
TP = c[1][1]

Acc=(TP+TN)/(TN+FP+FN+TP)
sens = TP/(TP+FP)
error = (FP+FN)/(TN+FP+FN+TP)
prediction = TP/(TP+FN)

print("\nconfusion matrix :- \n",c)
print('Accuracy of SVC=',round(Acc*100,4),'%')
print('Sensitivity of SVC=',round(sens*100,4),'%')
print('Error of SVC=',round(error*100,4),'%')
print('Prediction of SVC=',round(prediction*100,4),'%')

#%%
print("\n\nReduction Technique : PCA")
print('Algorithm = SVC\n' )
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

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(X_train, Y_train)
# score = svc_classifier.score(X_test, Y_test)


y_pred = svc.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
c = confusion_matrix(Y_test, y_pred)
TN = c[0][0]
FP = c[0][1]
FN = c[1][0]
TP = c[1][1]

Acc=(TP+TN)/(TN+FP+FN+TP)
sens = TP/(TP+FP)
error = (FP+FN)/(TN+FP+FN+TP)
prediction = TP/(TP+FN)

print("\nconfusion matrix :- \n",c)
print('Accuracy of SVC=',round(Acc*100,4),'%')
print('Sensitivity of SVC=',round(sens*100,4),'%')
print('Error of SVC=',round(error*100,4),'%')
print('Prediction of SVC=',round(prediction*100,4),'%')
#%%
print("\n\nReduction Technique: LDA")
print('Algorithm = SVC\n' )
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

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(X_train, Y_train)
# score = svc_classifier.score(X_test, Y_test)


y_pred = svc.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
c = confusion_matrix(Y_test, y_pred)
TN = c[0][0]
FP = c[0][1]
FN = c[1][0]
TP = c[1][1]

Acc=(TP+TN)/(TN+FP+FN+TP)
sens = TP/(TP+FP)
error = (FP+FN)/(TN+FP+FN+TP)
prediction = TP/(TP+FN)

print("\nconfusion matrix :- \n",c)
print('Accuracy of SVC=',round(Acc*100,4),'%')
print('Sensitivity of SVC=',round(sens*100,4),'%')
print('Error of SVC=',round(error*100,4),'%')
print('Prediction of SVC=',round(prediction*100,4),'%')
#%%
print("n\nReduction Technique: MLE")
print('Algorithm = SVC\n' )
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

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(X_train, Y_train)
# score = svc_classifier.score(X_test, Y_test)


y_pred = svc.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
c = confusion_matrix(Y_test, y_pred)
TN = c[0][0]
FP = c[0][1]
FN = c[1][0]
TP = c[1][1]

Acc=(TP+TN)/(TN+FP+FN+TP)
sens = TP/(TP+FP)
error = (FP+FN)/(TN+FP+FN+TP)
prediction = TP/(TP+FN)

print("\nconfusion matrix :- \n",c)
print('Accuracy of SVC=',round(Acc*100,4),'%')
print('Sensitivity of SVC=',round(sens*100,4),'%')
print('Error of SVC=',round(error*100,4),'%')
print('Prediction of SVC=',round(prediction*100,4),'%')
