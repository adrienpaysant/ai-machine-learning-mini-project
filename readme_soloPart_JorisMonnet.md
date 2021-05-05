## Decision tree 

# 1. Introduction

J'ai décidé pour cette partie du TP d'utiliser les decisions tree. C'est une technologie que j'avais déjà mis en place dans un précedent TP dont je suis reparti pour réaliser celui-ci.

# 2. Implémentation

Dans un premier temps, je récupère X et y de la même manière que pour la partie commune. Et je split en X_train, X_test, y_train et y_test.

Ensuite je crée le model avec DecisionTreeClassifier de sklearn. et je fit les données dedans.

Ensuite, je récupère la prediction de X_test de dans la variable y_pred. je plot la matrice de confusion et je crée un pdf avec le decision tree via ma fonction show_decision_tree.

Dans un second temps, j'utilise un grid search avec en paramètre, la profondeur maximum (15 pour ne pas affecter trop les performances) et l'état aléatoire entre 1 et 100. 

A partir du model en gridsearch, je fit X_train et y_train.

De ce fit, je récupère le meilleur model et je prédit comme plus haut le X_test. 

Enfin, j'affiche le classification_report, la matrice de confusion et je génère le pdf pour cette partie.


# 3. Résultats

Notre modèle donne environ 99% de bonnes réponses donc le modèle est considéré comme fonctionnel et bon. La matrice de confusion montre que le modele trouve 54 labels au mauvais endroit pour 3928 predictions.

