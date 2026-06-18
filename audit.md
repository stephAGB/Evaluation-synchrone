# Audit MLOps du projet churn

## Défaut 1 — Configuration du token en dur 
**Localisation** : app.py ligne 12
**Description** : La variable d'environnement "API_TOKEN" est hardcodée dans le fichier app.py. Ce n'est pas une bonne pratique car cela compromet la sécurité du projet.
**Criticité** : HAUTE
**Justification** : N'importe qui peut appeler l'endpoint /predict. De plus, coder des secrets ou des configurations en dur empêche de changer facilement d'environnement (dev, staging, prod) sans modifier le code. On pourrait utiliser un fichier .env et la librairie python-dotenv pour gérer les variables d'environnement. 


## Défaut 2 — Préccision des requirements
**Localisation** : requirements.txt, ligne 1 - 9
**Description** : Les requirements ne sont pas assez précis. Il est impossible de savoir exactement ce que le projet attend.
**Criticité** : MOYENNE
**Justification** : Les requirements ne disent pas quelle version est nécessaire pour chaque librairie, ce qui rend difficile la mise en place d'un environnement reproductible. De plus, on ne sait pas si l'ensemble du projet est couvert par ces requirements. 
Il faudrait ajouter des commentaires pour clarifier les intentions de l'auteur et les dépendances manquantes. Il faudrait aussi expliquer comment utiliser ces requirements dans le readme.


## Défaut 3 - .gitignore incomplet
**Localisation** : .gitignore
**Description** : Le fichier .gitignore n'est pas complet. Il prend en compte  le dossier qui devraient être ignorés.
**Criticité** : HAUTE
**Justification** : Le dossier data n'est pas ignoré, ce qui pourrait exposer des informations sensibles. De plus ajouter les données dans git n'est pas une bonne pratique car les fichiers de données peuvent être trop lourds. 
Il faudrait utiliser un gestionnaire de données comme DVC.


## Défaut 4 - Artefacts du modèle non versionnés
**Localisation** : main.py
**Description** : Les artefacts du modèle ne sont pas versionnés. Dans le code le modèle est entrainé et loggé via mlflow mais il n'est pas versionné via le model registry. De plus il n'est pas réellement enregistré dans le dossier artifacts.
**Criticité** : HAUTE
**Justification** : Les artefacts du modèle ne sont pas versionnés, ce qui rend difficile la mise en place d'un environnement reproductible. Et le retour en arrière si un probleme était détecté sur le modèle actuel ne serait pas possible car il n'y a aucune version disponible.

## Défaut 5 - Absence de métriques d'observabilité
**Localisation** : app.py lignes 55-57 et 16-19
**Description** : Les métriques sont initialisées mais pas réellement utilisées, donc il n'y aucun moyen de suivre les performances du modèle et le nombre de requêtes, de prédictions, d'erreurs, etc. 
**Criticité** : HAUTE
**Justification** : Il n'y a pas de moyen de visualiser les métriques. Elles sont d'ailleurs très limitées (uniquement pour le nombre d'erreurs et de prédictions) et ne permettent donc pas de savoir par exemple si le modèle commence à dériver ou si la qualité de ses prédictions diminue. 
Il faudrait mettre en place des métriques plus complètes et les logger.


## Défaut 6 - Logs insuffisants
**Localisation** : app.py  
**Description** : dans le pipeline d'entrainement il n'y a aucun log pour suivre la progression ou le temps d'exécution des étapes.
Dans l'API, les logs d'inférence se limitent à prediction=.... 
**Criticité** : MOYENNE
**Justification** : En l'absence des logs, il est difficile de savoir, si le pipeline s'est déroulé normalement, si des erreurs se sont produites et lesquelles, etc, ce qui augmenterait considérablement le temps de debuggage en cas de problème.
Il faudrait mettre en place des logs plus complets.
