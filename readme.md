# configuration de ollama

Sur la machine hôte, il y a quelques commandes à lancer pour configurer Ollama.  
Tout d'abord :
```
sudo systemctl edit ollama.service
```  
Ca ouvre les paramètres de Ollama, il faut éditer le fichier de façon à avoir ceci :
```
[Service]
<autres variables déjà présentes>
Environment="OLLAMA_HOST=0.0.0.0" #0.0.0.0 signifie que le serveur sera accessible depuis n'importe quelle interface réseau de la machine
Environment="OLLAMA_NUM_PARALLEL=2" #Cette variable définit le nombre de requêtes parallèles (simultanées) qu'Ollama peut traiter en même temps.
```
Sauvegardez les modifications puis redémarrer Ollama avec cette commande :
```
sudo systemctl restart ollama
```

~~je n'ai pas encore trouvé comment utiliser seulement 1 modèle vlm et plusieurs llm~~  
on va utiliser ce modèle pour tous les types de queries : mistral-small3.2:24b.  
ça va permettre d'éviter de swap les modèles en mémoire et on peut limiter l'utilisation du endpoint d'analyse d'image.

**Redis** a été mis en place, ça permet d'avoir une mémoire de la conversation basée sur un token de session id (actuellement désactivé).
Il faut maintenant trouver comment en avoir 1/utilisateur => login (vérifier aussi comment la mémoire se comporte, idem pour le stockage)  
Voir aussi comment on peut utiliser KV et la quantisation pour le modèle llm