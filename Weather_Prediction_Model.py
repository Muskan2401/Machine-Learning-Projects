#!/usr/bin/env python
# coding: utf-8

# # Weather Prediction Model

# In[10]:


#Importing the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[11]:


#Importing data
data=pd.read_csv(r'C:\Users\muska\OneDrive\Documents\Indian weather data\weather data.csv')
data


# In[12]:


data.describe()


# In[13]:


#Checking for null values
data.isnull==True


# # Linear Regression

# #### Visualization

# In[14]:


sns.scatterplot(x="HeatIndexC",y="WindChillC",data=data)
plt.title('WindChill vs HeatIndex',size=20)
plt.ylabel('WindChillC', size=12)
plt.xlabel('HeatIndexC', size=12)
plt.show()


# #### From the above scatter plot it can be noted that there is a correlation between heatindex and windchill

# In[15]:


X=np.array(data["HeatIndexC"])
Y=np.array(data["WindChillC"])
X=X.reshape(-1,1)


# In[16]:


#Splitting the dataset into training and test set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


#Feature scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[17]:


#CREATING A MODEL
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#Training the model
regressor.fit(X_train,Y_train)


# In[18]:


#Using the training dataset for prediction
pred=regressor.predict(X_test)


# In[19]:


#Model performance
from sklearn.metrics import r2_score,mean_squared_error
mse=mean_squared_error(Y_test,pred)
r2=r2_score(Y_test,pred)#Best fit line
plt.plot(X_test,pred,color='Black',marker='o')

#Results
print("Mean Squared Error : ", mse)
print("R-Squared :" , r2)
print("Y-intercept :"  , regressor.intercept_)
print("Slope :" , regressor.coef_)


# # NAIVE BAYES 

# In[20]:


#creating the dataset
X=data.loc[:300,["cloudcover","humidity"]].values
Y=data.loc[:300,"uvIndex"].values


# In[21]:


#Splitting the dataset into training and testing set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

#Feature Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[22]:


#Training the Naive Bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)


# In[23]:


y_pred=classifier.predict(X_test)


# In[24]:


y_pred


# In[25]:


Y_test


# ### Visualization

# In[26]:


plt.scatter(X_test[:, 0], X_test[:, 1],c=y_pred, s=20, cmap='plasma')
plt.xlabel('cloudcover')
plt.ylabel('humidity')


# In[27]:


#Model Performance
from sklearn.metrics import accuracy_score
ac=accuracy_score(Y_test,y_pred)

#Results
print("Accuracy score: ",ac)


# # DECISION TREE

# In[28]:


#creating the dataset
df=pd.DataFrame(data.iloc[:260,[0,19,21,22,23,24,25]])


# In[29]:


df


# In[30]:


df['City name'].unique()


# ### Visualization

# In[31]:


sns.heatmap(df.corr())#correlation matrix


# In[32]:


#Let's plot pair plot to visualize the attributes all at once
sns.pairplot(data=df,hue='City name')


# In[33]:


#we will seperate the target variable(y) and features(x) as follows
target=df['City name']
df1=df.copy()
df1=df1.drop('City name',axis=1)
x=df1


# In[35]:


#Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
target=le.fit_transform(target)
y=target


# In[36]:


#Splitting the data into train-test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[37]:


#Defining decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)

#predicting the values of test data
y_pred=dtree.predict(x_test)


# In[318]:


#Model performance
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Results
print("Classification report \n",classification_report(y_test,y_pred))


# In[319]:


#Results using graph visualization
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidth=5,annot=True,square=True,cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title='Accuracy score {:0.3f}'.format(dtree.score(x_test,y_test))
plt.title(all_sample_title,size=15)


# In[322]:


#Visualizing the tree without the use of graphviz
from sklearn.tree import plot_tree
plt.figure(figsize=(20,20))
dec_tree=plot_tree(decision_tree=dtree,feature_names=df1.columns,class_names=['Bengaluru','Bombay','Delhi'],filled=True,precision=4,rounded=True)


# # SVM

# In[38]:


#creating the model

#Defining the attributes
x=data.iloc[:201,2:4]

#we will seperate the target variable(y) and features(x) as follows
target=pd.DataFrame(data.loc[:200,'City name'])
y=target
y['City name'].unique()


# In[39]:


#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)


# In[40]:


#Splitting the dataset into train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

#Defining SVM algorithm
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=1)
classifier.fit(x_train,y_train)

#Predicting the value
y_pred=classifier.predict(x_test)


# ### Visualization

# In[900]:


#Visualizing the training set
from matplotlib.colors import ListedColormap
plt.figure(figsize=(7,7))
x_set,y_set=x_train,y_train

X1,X2=np.meshgrid(np.arange(start=x_set.iloc[:,0].min()-1,stop=x_set.iloc[:,0].max()+1,step=0.01),np.arange(start=x_set.iloc[:,1].min()-1,stop=x_set.iloc[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('black','white')))
plt.xlim=(X1.min(),X1.max())
plt.ylim=(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set.iloc[y_set==j,0],x_set.iloc[y_set==j,1],c=ListedColormap(('red','orange'))(i),label=j)
plt.title('Bengaluru vs Bombay')
plt.xlabel('maxtempC')
plt.ylabel('mintempC')
plt.legend()
plt.show()


# In[901]:


#Visualizing the testing set/Predictions
from matplotlib.colors import ListedColormap
plt.figure(figsize=(7,7))
x_set,y_set=x_test,y_test

X1,X2=np.meshgrid(np.arange(start=x_set.iloc[:,0].min()-1,stop=x_set.iloc[:,0].max()+1,step=0.01),np.arange(start=x_set.iloc[:,1].min()-1,stop=x_set.iloc[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75, cmap=ListedColormap(('black','white')))
plt.xlim=(X1.min(),X1.max())
plt.ylim=(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set.iloc[y_set==j,0],x_set.iloc[y_set==j,1],c=ListedColormap(('red','orange'))(i),label=j)
plt.title('Bengaluru vs Bombay predictions')
plt.xlabel('maxtempC')
plt.ylabel('mintempC')
plt.legend()
plt.show()


# In[908]:


#Model performance
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
accuracy=float(cm.diagonal().sum())/len(y_test)

#Results
print("Accuracy of SVM for the given dataset: ",accuracy)
print("Confusion matrix of SVM for the given dataset:\n ",cm)


# # Multiple Logistic Regression 

# In[49]:


data.columns


# In[50]:


#Creating the model
df=data.loc[:191,]
df=df.rename(columns={'City name':'target'})
df


# In[51]:


#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['target']=le.fit_transform(df['target'])


# In[52]:


df.info()


# In[53]:


#We'll drop the columns with date and time
df=df.drop(['date_time','moonrise','moonset','sunrise','sunset'],axis=1)

#Seperating numeric and categorical columns
numeric_cols=['DewPointC','WindGustKmph','cloudcover','humidity','visibility']
cat_cols=['target']

print(numeric_cols)
print(cat_cols)


# In[54]:


#Splitting the dataset into train test set 
random_seed=888

df_train,df_test=train_test_split(df,test_size=0.2,random_state=random_seed,stratify=df['target'])

print(df_train['target'].value_counts(normalize=True))
print()
print(df_test['target'].value_counts(normalize=True))


# In[55]:


#Transform the numerical variables:Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(df_train[numeric_cols])


def get_features_and_target_arrays(df,numeric_cols,cat_cols,sc):
    X_numeric_scaled=sc.transform(df[numeric_cols])
    X_categorical=df[cat_cols].to_numpy()
    X=np.hstack((X_categorical,X_numeric_scaled))
    Y=df['target']
    return X,Y

x,y=get_features_and_target_arrays(df_train,numeric_cols,cat_cols,sc)


# In[56]:


#Fit the logistic regression model
from sklearn.linear_model import LogisticRegression

#Fit the Logistic Regression Model
classifier=LogisticRegression(penalty='none')
classifier.fit(x,y)

#Evaluate the model
x_test,y_test=get_features_and_target_arrays(df_test,numeric_cols,cat_cols,sc)


# In[59]:


#Model performance
from sklearn.metrics import log_loss,plot_roc_curve,plot_confusion_matrix,plot_precision_recall_curve,roc_auc_score,recall_score,precision_score,average_precision_score,f1_score,classification_report,accuracy_score

plot_roc_curve(classifier,x_test,y_test)


# In[60]:


plot_precision_recall_curve(classifier,x_test,y_test)


# In[62]:


#Results
test_prob = classifier.predict_proba(x_test)[:, 1]
test_pred = classifier.predict(x_test)

print('Log loss = {:.5f}'.format(log_loss(y_test, test_prob)))
print('AUC = {:.5f}'.format(roc_auc_score(y_test, test_prob)))
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, test_prob)))
print('\nUsing 0.5 as threshold:')
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, test_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, test_pred)))
print('Recall = {:.5f}'.format(recall_score(y_test, test_pred)))
print('F1 score = {:.5f}'.format(f1_score(y_test, test_pred)))

print('\nClassification Report')
print(classification_report(y_test, test_pred))
plot_confusion_matrix(classifier, x_test, y_test)


# # K means clustering

# In[63]:


#Visualization of the dataset
plt.scatter(data[:701]['humidity'],data[:701]['cloudcover'])
plt.show()


# In[65]:


x=data.loc[:699,['humidity','cloudcover']]
#Fitting the algorithm 
from sklearn.cluster import KMeans
kmeans=KMeans(4)
kmeans.fit(x)

#Predicting the clusters
identified_clusters=kmeans.fit_predict(x)


# In[891]:


data_with_clusters=data[:700].copy()
data_with_clusters['Clusters']=identified_clusters
plt.scatter(data_with_clusters['humidity'],data_with_clusters['cloudcover'],c=data_with_clusters['Clusters'],cmap='plasma')


# ## Hierarchical Agglomerative clustering

# In[8]:


#Creating the model
X = data.iloc[:400, [2, 3]].values

#Plotting dendrogram for visualization
import scipy.cluster.hierarchy as sch
dendro = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('tempC')
plt.ylabel('visibility')
plt.show()

#Applying HAC and predicting the values
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of cities')
plt.xlabel('tempC')
plt.ylabel('visibilty')
plt.legend()
plt.show()


# In[ ]:




