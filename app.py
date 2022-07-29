# import libraries
from pkgutil import get_data
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app heading
st.write('''
# Explore different ML models and datasets
lets see which models are best?
''')

# dataset k name se aik box main dal k sidebar main laga do

dataset_name=st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# classifier name and the section

classifier_name=st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

#defining import dataset function

def get_dataset(dataset_name):
    data=None
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y


# ab hum x,y variable ko data assign kreingay


X, y=get_dataset(dataset_name)

#showing the shape of dataset
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))


# next hum different classifier k parameter ko userinput main add krein gay
def add_parameter_ui(classifier_name):
    params=dict() #creating an empty dictionary
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C # its the degree of correct classification
    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,15)
        params['K']=K # its the name of nearest neighbour
    else:
        max_depth=st.sidebar.slider('max_depth', 2,15)
        params['max_depth']=max_depth #depth of every tree that grow in random forest
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators #number of trees
    return params


params=add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    clf=None
    if classifier_name=='SVM':
        clf=SVC(C=params['C'])
    elif classifier_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf=clf=RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],random_state=1234)
    return clf

# ab iss function ko call krein gay or clf variable ko assign kr deingay

clf=get_classifier(classifier_name, params)

# ab hum dataset ko test or train main split krein gay by 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# ab hum ne apne classifier ki training krni hai

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

# checking model's accuracy score

acc=accuracy_score(y_test,y_pred)
st.write(f'classifier = {classifier_name}')
st.write(f'accuracy = {acc}')

### Plot dataset ###
# ab hum apne sare features ko 2-Dimentional plot main draw krein gay using pca
pca=PCA(2)
X_projected=pca.fit_transform(X)

# ab hum apna data 0 or 1 dimention main slice kr dein gay
x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar()

st.pyplot(fig)

