# Opérations et reprise du calcul figé

Notebook réellement créé par Kaggle :
https://www.kaggle.com/code/mahrouch/c1-e43-inceptiontime-adapted-private-gpu-pilot

L'API a construit le slug à partir du titre (en ajoutant `adapted`). Le code
scientifique de la version 1 reste celui du commit
`22f0ccaa0597949a94c347475394abddd3e82967`, tag `C1-E43-v1-frozen`.
Le dossier `kaggle/launch_v1/` conserve le notebook envoyé, sa configuration,
ses empreintes et les métadonnées reçues du serveur.

Le futur `BuildKaggle` exporte HEAD : il prépare un NOUVEAU calcul et ne doit
pas être employé pour reprendre les checkpoints v1. Pour reprendre v1, utiliser
le notebook figé, vérifier ses empreintes, conserver son code/configuration,
attacher l'output précédent E43 (`c1e43_resume.zip` et reçu) comme source de
notebook, puis sauvegarder/exécuter une nouvelle version sur le même type de GPU.
Ne jamais attacher les checkpoints ResNet pour poursuivre InceptionTime.
La vérification d'identité refuse les différences de code, données ou runtime.

Les pauses de budget sont prévues ; elles n'ouvrent pas automatiquement une
nouvelle session. Après une interruption forcée, conserver les fichiers et
examiner l'état avant reprise. Les checkpoints sont atomiques à l'époque ; le
ZIP exporté est renouvelé à chaque job terminé et à la sortie normale.

Pour consulter le statut avec le client Kaggle déjà authentifié :

```powershell
kaggle kernels status mahrouch/c1-e43-inceptiontime-adapted-private-gpu-pilot
```

Pour télécharger les sorties d'une version terminée, préciser son numéro :

```powershell
kaggle kernels output mahrouch/c1-e43-inceptiontime-adapted-private-gpu-pilot/1 -p outputs_received
```

Vérifier SHA-256 du ZIP contre `c1e43_resume.receipt.json`, l'expérience C1-E43,
le commit/configuration et les reçus de chaque job avant analyse scientifique.
La complétude attendue est 45 jobs et 75 jeux de prédictions. L'agrégation
locale par `c1inception.cli analyze` exige toutes les partitions/graines et
compare aux 24 pipelines E32 sans choisir leur meilleur score.

## Reprise du 8 novembre 2026 — version 2

La version 1 est COMPLETE au sens Kaggle, mais SESSION_PAUSED au sens de
l'expérience : le budget prévu a été atteint. Audit sauvegardé dans
`provenance/resume_v1_audit_20261108.json` : archive SHA-256 conforme au reçu,
28 jobs complets, 58 jeux de prédictions et 424 artefacts de jobs vérifiés.
Le job 29 (AWGN20, pli 5, graine 43) a un checkpoint de refit à l'époque 2/3,
avec modèle, optimiseur et RNG sauvegardés ; 16 jobs n'ont pas encore commencé.

`tools/prepare_resume.py` construit un notebook de reprise autour de la cellule
scientifique v1 strictement inchangée. Une cellule préalable exige exactement
l'archive auditée ; absence, duplication ou mauvaise empreinte provoquent un
arrêt avant tout entraînement. Ce contrôle a été essayé sur l'archive réelle,
un dossier sans checkpoint et une empreinte incorrecte. Les sorties v1 sont
attachées comme source de notebook ; aucun nouveau dataset n'est téléversé.
Les fichiers envoyés/reçus sont conservés dans `kaggle/resume_v2_20261108/`.

Le code scientifique reste `22f0ccaa0597949a94c347475394abddd3e82967`.
Les nouveaux outils d'audit/préparation/lecture de journaux n'altèrent pas le
runner embarqué, l'architecture, le budget ou les données. Le statut courant
est dans `research_record/launch_status.json` ; les preuves de reprise et de
saut des jobs terminés sont conservées dans `provenance/`.
