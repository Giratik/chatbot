## Version en date du 17/06/2026
### Page chatbot RH déplacée
La page du chatbot dédiée aux documents RH a été déplacée [ici](http://10.75.12.10:8503)


## Version en date du 05/06/2026
### Combinaison des pages
La page principale peut maintenant traiter les fichiers excel comme le faisait la page dédiée précédemment.  
Il est conseillé de recharger la page après avoir fini de traiter un fichier excel pour libérer la mémoire, dans le cas ou vous voudriez continuer à discuter.

### Sauvegarde des conversations
Il est maintenant possible d'exporter sa conversation dans un fichier à télécharger pour plus tard la charger dans le chatbot et reprendre la conversation.


## Version en date du 20/05/2026
### Modèle IA
Puisque le modèle d'IA utilisé à la capacité de raisonner, littéralement capable de s'interroger sur ce qu'il va répondre, on donne le choix à l'utilisateur d'activer la fonctionnalité car le mode raisonnement augmente le temps de génération de réponse proportionnellement à la demande.


## Version en date du 19/05/2026
### Changement de modèle IA
On passe de ministral-3:14b (modèle de mistral) à gemma4:e4b (modèle de google) toujours en local.
Gain de perfomances phénoménale en échange de quleques secondes de latence supplémentaires :
- On passe de 12288 token de contexte à 30000
- On peut augmenter le nombre d'instance en parallèle

Tout ça en consommant à peine la moité de la mémoire GPU


## Version en date du 18/05/2026
### Page Changelog
Création de la page changelog.

### Page assistant excel
L'outil peut maintenant traiter les gros fichiers de plusieurs milliers de lignes.  
La vue des fichiers a été déplacée dans la marge à gauche.  
Il n'y a plus besoin de sélectionner le type de raisonnement dans la marge, questionner le fichier et demander des graphes se font au même endroit.