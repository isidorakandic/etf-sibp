#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas.util.testing as tm

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn import tree


# ### učitavanje podataka

# In[9]:


data = pd.read_csv('./Kannada-MNIST/train.csv')
print("Labele: ", data['label'].unique())


# In[3]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.groupby(by='label').size()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], 
                                                    data.iloc[:, 0], 
                                                    test_size=0.3,
                                                    stratify=data['label'])


# In[5]:


print('Train Shape: ', X_train.shape)
print('Test Shape:', X_test.shape)


# In[11]:


# provera da li raspodela labela uniformna
y_train.value_counts()
y_test.value_counts()


# ### primer Kannada MNIST slika

# In[11]:


fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))

for i in range(10):
    num_i = X_train[y_train == i]
    ax[0][i].set_title(i)
    for j in range(10):
        ax[j][i].axis('off')
        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), 
                        cmap='gray')


# ## algoritmi klasifikacije

# ### 1. Logistička regresija

# In[44]:


from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(multi_class='multinomial', max_iter=200)

parameters = {'C':(1,10), 'penalty':('l1','l2')}
LR_models = GridSearchCV(LR_model, parameters)
LR_models.fit(X_train, y_train)


# In[46]:


print('Optimalni parametri za logističku regresiju su', LR_models.best_params_)


# In[48]:


y_predicted_LR = LR_models.best_estimator_.predict(X_test)
LR_score = accuracy_score(y_predicted_LR, y_test)
LR_cm = confusion_matrix(y_test, y_predicted_LR)


# In[49]:


print('Preciznost logističke regresije je', LR_score)


# In[50]:


print(classification_report(y_test, y_predicted_LR))


# In[51]:


plt.figure(figsize = (9,9))
sns.heatmap(LR_cm, 
            annot = True, 
            fmt = "d", 
            linewidths = .5, 
            square = True, 
            cmap = 'Reds_r');

plt.ylabel('Tačna labela');
plt.xlabel('Prediktovana labela');
title = 'Matrica konfizuje za linearnu regresiju'
plt.title(title, size = 15)


# ### 2. Drvo odlučivanja

# In[52]:


from sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier()
parameters = {'criterion':('gini', 'entropy')}

DT_models = GridSearchCV(DT_model, parameters)
DT_models.fit(X_train, y_train)


# In[53]:


print('Optimalni parametri za drvo odlučivanja su', DT_models.best_params_)


# In[54]:


y_predicted_DT = DT_models.best_estimator_.predict(X_test)
DT_score = accuracy_score(y_predicted_DT, y_test)
DT_cm = confusion_matrix(y_test, y_predicted_DT)


# In[55]:


print('Preciznost drveta odlučivanja je', DT_score)


# In[56]:


print(classification_report(y_test, y_predicted_DT))


# In[57]:


plt.figure(figsize = (9,9))
sns.heatmap(DT_cm, 
            annot = True, 
            fmt = "d", 
            linewidths = .5, 
            square = True, 
            cmap = 'Blues_r');

plt.ylabel('Tačna labela');
plt.xlabel('Prediktovana labela');
title = 'Matrica konfuzije za drvo odlučivanja'
plt.title(title, size = 15)


# In[64]:


import graphviz 

def save_tree(model):
    tree.plot_tree(model) 
    dot_data = tree.export_graphviz(model, out_file = None) 
    graph = graphviz.Source(dot_data)
    graph.render("decisiontree_depth4") 

DT_plot_model = DecisionTreeClassifier(max_depth=4)
DT_plot_model.fit(X_train, y_train)
y_predicted_DT_plot = DT_plot_model.predict(X_test)
DT_plot_score = accuracy_score(y_predicted_DT_plot, y_test)
DT_plot_cm = confusion_matrix(y_test, y_predicted_DT_plot)


# In[65]:


save_tree(DT_plot_model)


# ### 3. Principal component analysis + Support Vector Machine

# In[66]:


from sklearn import svm
from sklearn.decomposition import PCA

pca = PCA(n_components=0.9, whiten = True)

X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)


# In[69]:


print('Broj komponenti nakon PCA tranformacije je', pca.n_components_)


# In[70]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(range(pca.n_components_), pca.explained_variance_ratio_)
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('Explained variance ratio by principal component')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('Cummulative explained variance ratio')
plt.grid()
plt.tight_layout()
plt.show()


# In[74]:


X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)

#SV_model = svm.SVC()
#parameters = {'kernel':('linear', 'poly', 'rbf')}
#SV_models = GridSearchCV(SV_model, parameters)
#SV_models.fit(X_train_PCA, y_train)

SV_model_linear = svm.SVC(kernel='linear')
SV_model_poly4 = svm.SVC(kernel='poly')
SV_model_rbf = svm.SVC(kernel='rbf')

#SV_model_linear.fit(X_train_PCA, y_train)
#y_predicted_SV_lin = SV_model_linear.predict(X_test_PCA)
#SV_score_lin = accuracy_score(y_predicted_SV_lin, y_test)
#SV_cm_lin = confusion_matrix(y_test, y_predicted_SV_lin)
#print("Linear SVM: ", SV_score_lin)

#SV_model_poly4.fit(X_train_PCA, y_train)
#y_predicted_SV_poly = SV_model_poly.predict(X_test_PCA)
#SV_score_poly = accuracy_score(y_predicted_SV_poly, y_test)
#SV_cm_poly4 = confusion_matrix(y_test, y_predicted_SV_poly)
#print("Poly SVM: ", SV_score_poly)

SV_model_rbf.fit(X_train_PCA, y_train)
y_predicted_SV_rbf = SV_model_rbf.predict(X_test_PCA)
SV_score_rbf = accuracy_score(y_predicted_SV_rbf, y_test)
SV_cm_rbf = confusion_matrix(y_test, y_predicted_SV_rbf)
print("RBF SVM: ", SV_score_rbf)

# SV_model.fit(X_train_PCA, y_train)
# y_predicted_SV = SV_model.predict(X_test_PCA)
# SV_score = accuracy_score(y_predicted_SV, y_test)
# SV_cm = confusion_matrix(y_test, y_predicted_SV)


# In[82]:


#print('Optimalni parametri za metod potpornih vektora (SVM) su', SV_models.best_params_)
y_predicted_SV = y_predicted_SV_rbf
SV_score = SV_score_rbf
SV_cm = SV_cm_rbf


# In[77]:


SV_cm_rbf = confusion_matrix(y_test, y_predicted_SV_rbf)


# In[78]:


print('Preciznost metoda potpornih vektora (SVM) je ', SV_score)


# In[79]:


print(classification_report(y_test, y_predicted_SV))


# In[83]:


plt.figure(figsize = (9,9))
sns.heatmap(SV_cm, 
            annot = True, 
            fmt = "d", 
            linewidths = .5, 
            square = True, 
            cmap = 'Greens_r');

plt.ylabel('Tačna labela');
plt.xlabel('Prediktovana labela');
title = 'Matrica konfuzije za metod potpornih vektora (SVM)'
plt.title(title, size = 15)


# #### vizualizacija PCA instanci pomoću t-SNE algoritma

# In[84]:


from sklearn.manifold import TSNE
tsne = TSNE(random_state = 42, 
            n_components = 2,
            verbose = 0, 
            perplexity = 40, 
            n_iter = 250).fit_transform(X_train_PCA)


# In[82]:


y_subset = y_train
plt.scatter(tsne[:, 0], 
            tsne[:, 1], 
            s = 5, 
            c = y_subset,
            cmap='Spectral')

plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Visualizing Kannada MNIST through t-SNE', fontsize=12);
plt.savefig("BLA.png")


# ### poređenje uspešnosti modela

# In[25]:


models = pd.DataFrame({
    'Model': ['LogisticRegression','Decision Tree', 'SVM'],
    'Score': [LR_score, DT_score, SV_score]})
models.sort_values(by = 'Score', ascending = False)


# ### primena modela na validacioni set

# In[26]:


validation_data = pd.read_csv("./Kannada-MNIST/Dig-MNIST.csv")


# In[27]:


X_val = validation_data.drop('label', axis=1)
y_val = validation_data['label']


# In[42]:


y_val_predicted_LR = LR_model.predict(X_val)
LR_val_score = accuracy_score(y_val_predicted_LR, y_val)


# In[ ]:


y_val_predicted_DT = GS_models.best_estimator_.predict(X_val)
DT_val_score = accuracy_score(y_val_predicted_DT, y_val)


# In[75]:


X_val_PCA = pca.fit_transform(X_val)
X_val_PCA.shape
y_val_predicted_SV = SV_model_rbf.predict(X_val_PCA)
SV_val_score = accuracy_score(y_val_predicted_SV, y_val)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistička regresija','Drvo odlučivanja', 'Metod potpornih vektora (SVM)'],
    'Score': [LR_val_score, DT_val_score, SV_val_score]})
models.sort_values(by = 'Score', ascending = False)

