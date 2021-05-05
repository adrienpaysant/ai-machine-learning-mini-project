# Rapport Adrien Paysant


## Introduction
Il a été choisit de réaliser le modèle Naif Bayes.


## Implémentation
On réccupère les données via la méthode loadData() réalisée dans la partie commune du travail pratique.

On va utiliser le modèle Gaussien fournit par sklearn.naive_bayes : 


``` from sklearn.naive_bayes import GaussianNB ```

On évalue nos performances avec  : ```from sklearn import metrics```

On peut alors afficher le résultat via la matrice de confusion ainsi que les score macro et micro f1.

## Résultat

On obtient une accuracy de 89%, ce qui est correct.

Voici un rendu graphique : 
![result](/img/result.png)
