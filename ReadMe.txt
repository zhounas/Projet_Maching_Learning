Nom: HOUNAS Zehor Thilleli
Niveau: Master 1
N° d'étudiant:19009892

Objectif:
Créer un modéle qui aprend a jouer au jeu "trex".

Le Zip contiens:
	1) un fichier "dino.py" .
	2) un dossier "trex" qui represente la version du jeu trex en local.

Principe de fonctionnement:
 	on crée un premier modéle (non entrainé) a deux entrées:"la distance" et "le saut"  et on a en sortie booléen, afin de prédire si le jeux va se terminé ou non,
	nous générons un datasets de "n" paries et nous entrainons notre modéle avec ce dernier,
 	cette opération est reprise tant que la précision obtenu est inférieur a 95%,
	une fois cette précision atteinte, nous obtenons le modéle finale.

Le projet est inspiré: 
	1) du TP "Diabeties". 
	2) de la video "A.I. Learns to play Flappy Bird" : "https://www.youtube.com/watch?v=WSW-5m8lRMs&feature=share&fbclid=IwAR0eSZO_ocifiQNZo5yitZAJoOz0PDIIXRpiaIyb8YG9KVGxTIfpTKNJ8-M".
  	
   