#!/usr/bin/env python
# coding: utf-8

#  # Travaux pratiques - Arbres de décision _TIZZAOUI_Youva_M2_SDTS
On commence par importer les bons modules (DATA CAR) :
# In[66]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


# In[94]:


data = pd.read_csv('car.data')
data


# In[95]:


data['g'].value_counts()

Modifier le tableau en valeur Numerique afin d'implimenter le tableau:
# In[96]:


df = pd.DataFrame(data,columns=['a', 'b', 'c', 'd', 'e', 'f','g'])

#df['small']= df['small'].replace(['small'],1)
df['a'].replace(['low','med', 'high','vhigh'], [0,1,2,3], inplace =True)


df['b'].replace(['low','med', 'high', 'vhigh'], [0,1,2,3], inplace =True)
df['c'].replace(['5more','2'], [5, 2], inplace =True)
df['d'].replace(['more', '2'], [5, 2], inplace =True)
df['e'].replace(['small','med', 'big'], [0,1,2], inplace =True)
df['f'].replace(['low','med', 'high'], [0,1,2], inplace =True)
df['g'].replace(['unacc','acc', 'good', 'vgood'], [0,1,2,4], inplace =True)

selectionner les 6 premeier ligne du tableau 
# In[97]:


data6= pd.DataFrame(df, columns=['a', 'b', 'c', 'd', 'e', 'f'])
data6


# In[98]:


data1= pd.DataFrame(df, columns=['g'])
data1

Avant de construire le modèle, séparons le jeu de données en deux : 70% pour l’apprentissage, 30% pour le test.
# In[99]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data6, data1, train_size=0.7,random_state=0)

Nous pouvons désormais construire un arbre de décision sur ces données :
# In[100]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

Une fois l’apprentissage terminé, nous pouvons visualiser l’arbre, soit avec matplotlib en passant par la méthode plot_tree, soit avec l’outil graphviz (commande dot). Par exemple, avec matplotlib :
# In[101]:


tree.plot_tree(clf, filled=True)

Une fois le modèle construit, il est possible de l’utiliser pour la prédiction sur de nouvelles données :
# In[102]:


clf.predict(X_test)

On peut de cette façon calculer le score en test :
# In[103]:


clf.score(X_test, y_test)

Question 1 :
Changez les valeurs de parametres max_depth et min_samples_leaf. Que constatez-vous ?
***********************************************************************************************
Sur-apprentissage : parfois les arbres générés sont trop complexes et généralisent mal. Choisir des bonnes valeurs pour les paramètres profondeur maximale (max_depth) et nombre minimal d’exemples par feuille (min_samples_leaf) permet d’éviter ce problème.

# In[104]:


clf = tree.DecisionTreeClassifier(max_depth = 3)
clf.fit(X_train, y_train)


# In[105]:


clf.score(X_test, y_test)


# In[106]:


clf = tree.DecisionTreeClassifier(min_samples_leaf = 20)
clf.fit(X_train, y_train)


# In[107]:


clf.score(X_test, y_test)

# Question :
Le problème ici étant particulièrement simple, refaites une division
apprentissage/test avec 5% des données en apprentissage et 95% test.
# In[108]:


X_train, X_test, y_train, y_test = train_test_split(data6, data1, train_size=0.95,random_state=0)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# # III/ Affichage de la surface de décision

# In[115]:


import numpy as np
import matplotlib.pyplot as plt

# Paramètres
n_classes = 4
plot_colors = "bry" # blue-red-yellow
plot_step = 0.02

# Choisir les attributs longueur et largeur des pétales
pair = [2, 3]
data = data6.values
data11 =  data1.values
print(data1)
# On ne garde seulement les deux attributs
X = data[:, pair]
y = data11

# Apprentissage de l'arbre
clf = tree.DecisionTreeClassifier().fit(X, y)

# Affichage de la surface de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#plt.xlabel(X.feature_names[pair[0]])
#plt.ylabel(y.feature_names[pair[1]])
plt.axis("tight")

# Affichage des points d'apprentissage
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=y[i], cmap=plt.cm.Paired)
plt.axis("tight")
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()


# # Arbres de décision pour la régression¶

# In[111]:


from sklearn import tree

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])


# In[117]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

# Créer les données d'apprentissage
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()

fig = plt.figure(figsize=(12, 4))
fig.add_subplot(121)
plt.plot(X, y)
plt.title("Signal sinusoïdal pur")

# On ajoute un bruit aléatoire tous les 5 échantillons
y[::5] += 3 * (0.5 - np.random.rand(16))
fig.add_subplot(122)
plt.plot(X, y)
plt.title("Signal sinusoïdal bruité")


# In[118]:


# Apprendre le modèle
reg = DecisionTreeRegressor(max_depth=2)
reg.fit(X, y)

# Prédiction sur la même plage de valeurs
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = reg.predict(X_test)

# Affichage des résultats
plt.figure()
plt.scatter(X, y, c="darkorange", label="Exemples d'apprentissage")
plt.plot(X_test, y_pred, color="cornflowerblue", label="Prédiction", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Régression par un arbre de décision")
plt.legend()
plt.show()


# In[ ]:




