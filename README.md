## Ai-machine-learning-mini-project


# Rapport partie commune :

1. Introduction
Le but de la partie commune de ce TP est de mettre en place un réseau de neurone sur un DataSet de moteur.

Ce Dataset est récupéré depuis des fichiers .csv.

Pour réaliser notre réseau nous utilisons la partie keras de la bibliothèque tensorflow : https://keras.io

Implémentation
Nous utilisons une fonction loadData() pour récupérer deux tableaux numpy.

Ils correspondent au X et y que l'on utilise ensuite pour spliter la partie train et la partie test.

La fonction LoadData() permet de récupérer uniquement une partie des données qui nous serons utiles.

Dans un premier temps, nous lisons chaque fichier csv avec pandas comme montré dans l'exemple.

Pour une taille de fenetre donnée(50), nous récupérons toutes les données d'une fenêre, puis on garde uniquement le minimum de chaque colonne, la maximum, la moyenne, et l'ecart type.

Nous avons ainsi 12 champs sur lesquels travailler.

X est donc le tableau contenant ces 12 champs par fenetre. y contient une liste d'id des labels (balanced, unbalanced,electric_fault).

Une fois les données extraites, on split X et y avec 30% de test.

Ensuite on met à l'échelle le X_train et le X_test à l'aide du StandardScaler de sklearn. la fonction fit permet de caculer la moyenne et l'écrta type qui serons utilisés ensuite.

Ensuite on utilise la fonction transform qui permet de standardizer notre X_train et X_test en les centrant et mettant à l'échelle.

Une fois fait, nous utilisons Sequential de keras. Nous affichons une première fois l'objet model avec model.summary().

Nous ajoutons les objets dense à notre model. On y donne le paramètre activation(rectified linear unit activation et softmax), la forme (12 pour nos 12 champs), et un entier qui correspond à la dimension de l'espace d'output.

Ensuite, nous réaffichons le model et cette fois un tableau avace les Dense s'affiche

Le model est compilé avec l'optimizer rmsprop qui est celui par défaut. Le paramètre metrics est aussi spécifié, ce que l'on recherche est l'accuracy de notre model. Le paramètre loss permet de récupérer les pertes entre les labels et les prédictions. cette fonction est utilisé quand il y a qu'une classe de label : https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class

Nous continuons en fittant le model. On utilise 50 epochs et 64 batch_size, ces paramètres sont trouvés empiriquement. le batch_size par défaut à 32 ne suffisait pas. On valide les données avec la fonction to_categorical pour convertir un vecteur vers une matrice binaire : https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical

Enfin, nous affichons le résultat et on évalue notre modele.

Résultats
Avec une accuracy à 99,44%, notre modèle semble bien fonctionner.
