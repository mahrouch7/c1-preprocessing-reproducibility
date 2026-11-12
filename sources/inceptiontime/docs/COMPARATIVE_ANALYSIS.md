# CURRENT — plan d'analyse descriptive E43/E33/E32

L'auteur demande de préparer le code et la commande PowerShell ; le calcul
scientifique complet sera lancé par lui. La récupération E43 est vérifiée :
45 tâches, 75 jeux de prédictions, 646 artefacts et 28 reçus antérieurs inchangés.
Archive finale et manifeste contrôlés indépendamment ; aucun nouveau score
InceptionTime calculé pendant cette préparation.

## Périmètre et comparabilité

Deux modèles temporels adaptés (InceptionTime E43 et ResNet E33), trois graines
42/43/44 par modèle, cinq plis identiques et 24 pipelines ML non filtrés.
Les cinq cas d'évaluation sont propre partagé, AWGN20 propre-entraîné et
apparié, mixed10 propre-entraîné et apparié. La référence propre n'est comptée
qu'une fois. Scalers de référence : standard pour LR/kNN/SVM/MLP, aucun pour
RF/NB/DT/GB ; RMS6/RAW12/MULTI18 pour chaque classifieur. Aucune sélection du
meilleur concurrent, de la meilleure graine ou du meilleur scaler après test.

Le script exige les audits/manifeste/predictions intacts, les mêmes tenseurs
figés, partitions imbriquées et normalisation entraînement, ainsi que la même
pile GPU enregistrée. Les scores ResNet doivent reproduire les 15 références
par graine déjà vérifiées dans le manuscrit, à 1e-10 près. Les étiquettes et
décisions doivent correspondre aux identités de fenêtre et à argmax(probabilité).
L'audit natif SVM antérieur n'est pas remplacé par une nouvelle règle de décision.

## Estimation et comparaison

Les prédictions sont réunies hors pli sur les 1839 fenêtres, avec poids
inverse de l'effectif famille/classe. Les scores ne sont pas des moyennes de
Macro-F1 par pli. Pour les réseaux, conserver les trois scores et calculer la
moyenne et l'écart-type échantillonnal ; pas d'ensemble probabiliste ni de
sélection de graine. Cet écart-type ne constitue pas un intervalle de confiance.
La médiane/étendue ML décrit 24 pipelines différents, pas la variabilité d'un
même modèle. Les neuf croisements d'initialisations sont fournis ; l'égalité
des numéros de graines entre architectures ne prouve pas un couplage statistique.

Macro-F1 primaire ; BA, MCC, fausses alarmes saines, défauts manqués, Brier et
log loss complémentaires. Taux/F1/BA : différences en points de pourcentage ;
MCC et pertes restent dans leurs unités. Pas de nouveau bootstrap, p-value,
test, correction Holm ou extension automatique des 728 contrastes historiques.

Les matrices/classes, erreurs ABC/ABCG et taux conditionnels accompagnent les
scores globaux. Les quatre masses d'erreur partagent 100(1−BA), pas le Macro-F1.
Les comparaisons de désaccords utilisent les mêmes fenêtres pondérées ; les
masses « A seul correct − B seul correct » donnent une différence de BA.
Les effets d'exposition comparent apparié moins propre-entraîné pour chaque
pipeline/graine/condition. Les temps enregistrés restent descriptifs et ne
mesurent ni un délai de relais ni un budget matériel parfaitement contrôlé.

## Sorties et contrôles

`Analyze-Results.ps1` crée un dossier horodaté distinct dans le répertoire
externe configuré. Il écrit les OOF des réseaux, 17 tableaux CSV, les manifestes,
un reçu et `analyse_fr.md`. Le processus scientifique utilise NumPy 2.3.5 ;
les graphiques sont produits dans un processus séparé utilisant uniquement les
tables, afin de ne pas contaminer l'environnement scientifique avec NumPy graphique.

Trois figures candidates PDF/SVG/PNG : performances et taux conditionnels,
décomposition des erreurs, différences de rappel par classe. Titres, unités,
sources CSV et empreintes sont conservés. Leur contrôle visuel sur les PDF réels
reste PENDING après l'exécution auteur ; aucune intégration au manuscrit automatique.

Interprétation à discuter après les sorties : stabilité des tendances entre
architectures, avantage limité à une condition/graine, compromis Macro-F1/fausse
alarme, et persistance des erreurs ABC/ABCG. Ni supériorité universelle, ni
validation externe, ni nouveauté algorithmique ne sont présumées.
