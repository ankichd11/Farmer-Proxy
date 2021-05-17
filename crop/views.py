from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import pickle


def crop(request):
    return render(request, 'crop.html')

def submit(request):
    a = float(request.POST.get("nitrogen"))
    b = float(request.POST.get("phosphorus"))
    c = float(request.POST.get("pottasium"))
    d = float(request.POST.get("temperature"))
    e = float(request.POST.get("humidity"))
    f = float(request.POST.get("ph"))
    g = float(request.POST.get("rainfall"))
    '''    
    df = pd.read_csv('Crop_data.csv')
    print(df.head())
    print(df.shape)
    print(df.info())
    target = ['label']
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

    models = []
    models.append(('LogisticRegression', LogisticRegression(random_state=0)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(random_state=0)))
    models.append(('XGBClassifier', XGBClassifier(random_state=0)))
    models.append(('GradientBoostingClassifier', GradientBoostingClassifier(random_state=0)))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('RandomForestClassifier', RandomForestClassifier(random_state=0)))

    model_name = []
    accuracy = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_name.append(name)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        print(name, metrics.accuracy_score(y_test, y_pred))

    clf = RandomForestClassifier(n_estimators=100)

    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)
    '''
    clf = pickle.load(open('crop_model', 'rb'))
    res = clf.predict([[a, b, c, d, e, f, g]])
    return render(request, 'crop.html', {"res":res})