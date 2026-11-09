# Module 3 — Embeddings, index et recherche (EF08, EF09, EF11)

Ce module transforme chaque passage en vecteur, range les vecteurs dans un index FAISS et retrouve les passages les plus proches d'une question. La génération de la réponse (citations, abstention) est le module 4.

Les choix viennent des décisions D4 et D8 à D12, détaillées dans [decisions.md](decisions.md).

## Le chemin d'un passage, puis d'une question

```
passages (module 2)
  │ indexation.py   texte encodé = « passage: » + chemin de section + contenu
  │                 empreinte (passages + modèle + révision + options) identique ?
  │                   oui → index rechargé, modèle non chargé            (EF09, D12)
  │                   non → raison journalisée, puis reconstruction :
  │ encodeur.py         longueur encodée vérifiée contre la limite du modèle  (D11)
  │                     vecteurs calculés par sentence-transformers          (EF08, D8)
  ▼ vectoriel.py        index FAISS exact, enregistré avec ses métadonnées   (D9)

question
  │ recherche.py    « query: » + question → vecteur → k passages les plus proches  (EF11)
  ▼                 score décroissant, ex æquo départagés par passage_id
```

`rag ingest` enchaîne import, découpage et indexation. `rag ask` interroge l'index ; pour l'instant, la commande affiche les passages retrouvés et leurs scores.

## Les fichiers, un par un

### `configs/default.yaml` et `src/rag/config.py` — sections `embeddings` et `recherche`

| Réglage | Défaut | Rôle | Dans l'empreinte |
|---|---|---|---|
| `embeddings.modele` | `intfloat/multilingual-e5-small` | modèle (EF08) | oui |
| `embeddings.revision` | `main` | hash du commit Hugging Face, à figer | oui (révision résolue) |
| `embeddings.prefixe_requete` | `"query: "` | préfixe e5 des questions | oui |
| `embeddings.prefixe_passage` | `"passage: "` | préfixe e5 des passages | oui |
| `embeddings.normaliser` | `true` | vecteurs de norme 1 : score = cosinus | oui |
| `embeddings.taille_lot` | 32 | textes encodés à la fois | non |
| `decoupage.contexte_section` | `true` | chemin de section en tête du texte encodé (D5) | oui |
| `recherche.k` | 5 | passages renvoyés (EF11) | — |

`configs/experiences/bge-m3-1024.yaml` est identique à la configuration par défaut, sauf la taille (1 024 tokens, chevauchement 128), le modèle (bge-m3) et les préfixes (vides : bge-m3 n'en utilise pas). C'est la configuration de la mesure de temps d'ENF02 (voir plus bas).

### `src/rag/index/encodeur.py`

- **Interface `Encodeur`.** Le reste du code ne dépend que d'elle : changer de modèle ne demande qu'une modification de configuration (critère d'EF08), et les tests utilisent un encodeur factice.
- **`EncodeurST`** enveloppe sentence-transformers (D8). Le dépôt du modèle sur Hugging Face décrit comment calculer le vecteur : moyenne des tokens pour e5, token initial pour bge-m3, puis normalisation. On ne le réécrit pas, car c'est l'endroit où une erreur passerait inaperçue.
- **`prompt=""` explicite.** Sans lui, sentence-transformers appliquerait le préfixe par défaut déclaré par le modèle, s'il en déclare un, et e5 recevrait `passage: passage: …` sans aucune erreur. Les préfixes sont ajoutés par notre code, une seule fois. J'ai vérifié ce comportement en lisant le code source de sentence-transformers 6.2, que je ne peux pas exécuter dans mon environnement. Le test `slow` `test_pas_de_prefixe_ajoute_par_sentence_transformers` le vérifie avec le vrai modèle.
- **`resoudre_revision`.** Elle donne le hash du commit sans charger les poids : un hash figé est renvoyé tel quel, sans réseau, et `main` est résolu via `config.json` du dépôt, avec un avertissement invitant à figer la révision. Grâce à elle, un index à jour se recharge sans charger le modèle.
- **`compter_tokens`.** Elle donne la longueur réellement encodée, tokens spéciaux compris (D11).

### `src/rag/index/vectoriel.py`

- **Index FAISS exact** (`IndexFlatIP`) sur des vecteurs normalisés : le score est le cosinus (D9). La recherche est exacte et instantanée à cette échelle : 2 657 passages, environ 11 Mo même en 1 024 dimensions.
- **Départage des ex æquo.** Score décroissant, puis `passage_id` croissant en cas d'égalité exacte, y compris au rang k. Sans ce départage, FAISS rendrait les ex æquo dans un ordre non garanti.
- **Enregistrement par sérialisation en mémoire** puis écriture par Python, au lieu de `faiss.write_index`, qui échoue sous Windows sur les chemins contenant des caractères non ASCII.

### `src/rag/index/indexation.py`

- **Empreinte (D12).** Elle couvre les passages (identifiant et texte exactement tel qu'il est encodé), le modèle, sa révision résolue et toutes les options d'encodage. Changer une seule de ces valeurs force la reconstruction, et la raison est journalisée : `revision : 'aaa…' → 'bbb…'`, `passages modifiés`, ou `aucun index pour cette configuration`.
- **Un dossier par configuration d'expérience.** `store/index/<config_decoupage>__<modèle>-<empreinte des options>/`. Deux configurations qui ne diffèrent que par une option ont chacune leur index et ne s'écrasent pas.
- **Troncature (D11).** La taille d'un passage (EF06) ne compte que son contenu. La longueur réellement encodée (préfixe, chemin de section, contenu, tokens spéciaux) est vérifiée à part, contre la limite du modèle : 512 pour e5, 8 192 pour bge-m3. Les passages qui la dépasseraient sont comptés, signalés et listés dans les métadonnées.
- **Contrôle de la normalisation.** Si la normalisation est demandée et qu'un vecteur n'a pas une norme de 1, l'indexation s'arrête en erreur.
- **Métadonnées** (`meta.json`) : empreinte, paramètres, modèle et révision, dimension, limite du modèle, passages tronqués, plus grande longueur encodée, durée d'encodage, passages par seconde, date, versions de torch, sentence-transformers, transformers, faiss-cpu et numpy.
- **`charger_index_a_jour`.** Utilisée par `rag ask`, elle refuse un index dont l'empreinte ne correspond plus : on n'interroge jamais un index périmé.

### `src/rag/index/recherche.py`

`MoteurRecherche.rechercher(question, k)` ajoute le préfixe de requête, encode la question avec le même modèle et renvoie les k passages les plus proches, chacun avec son rang et son score (EF11).

### `src/rag/cli.py`

- **`rag ingest`** ajoute une ligne : `Index … : construit (raison) · N passages en X s (Y passages/s) · D dimensions · T passage(s) tronqué(s)`, ou `rechargé en X s`. Une barre de progression s'affiche pendant l'encodage.
- **`rag ask "question" --config … [--k N]`** affiche le temps de rechargement de l'index, puis les passages retrouvés : rang, score, fichier, section (chemin raccourci à l'affichage) et extrait.

### `src/rag/fichiers.py`

L'écriture atomique d'un dossier, partagée par les passages et l'index : un arrêt en cours ne laisse jamais un mélange d'ancien et de nouveau contenu, et le procédé fonctionne sous Windows.

## Sur le vrai corpus

Mesuré sous Windows, sur CPU, sans GPU, avec la configuration par défaut (multilingual-e5-small, révision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, passages de 256 tokens) :

| Mesure | Valeur |
|---|---|
| Passages encodés | 2 657 |
| Durée d'encodage | 120,8 s, soit 22 passages par seconde |
| Dimension | 384 |
| Passages tronqués | 0, comme attendu à 256 tokens (D11) |
| Rechargement de l'index | instantané, sans charger le modèle |

C'est très en deçà des 30 minutes d'ENF02. La mesure avec bge-m3 à 1 024 tokens, beaucoup plus lourde, reste à faire (plus bas).

### Ce que le vrai modèle a montré

**Calcul des vecteurs (D8) : correct.** Les tests `slow` vérifient la norme 1 des vecteurs, l'absence de préfixe doublé et quatre comparaisons par paires, dont une question française face à des passages anglais. Tous passent.

**Recherche multilingue : une limite observée.** Un test plus strict, que j'avais ajouté, exigeait que chaque question classe son passage premier parmi les quatre. La question française « Comment rendre le stockage indépendant du cycle de vie des Pods ? » a classé premier le passage sur les Pods (0,824), devant celui sur les volumes (0,811). Le mot « Pods », présent dans la question, l'emporte sur le sens de la phrase française. Ce test est désormais limité aux questions anglaises, car il mesurait la qualité du modèle et non son calcul. L'observation est conservée : c'est exactement l'écart que le Recall@5 par langue (D2) devra quantifier au module 5, avec un jeu de questions françaises qui ne contiennent pas toutes le terme anglais cherché.

**Scores resserrés : à retenir pour l'abstention.** Pour « How do I create a Pod? », les 5 premiers passages ont des scores entre 0,878 et 0,886, et même le passage hors sujet du test ci-dessus obtenait 0,738. Les scores d'e5 occupent une plage étroite et élevée : un seuil d'abstention fixe et intuitif, comme 0,5, ne s'appliquerait jamais. Le seuil d'EF18 devra être calibré sur le jeu de test (module 4), et non choisi a priori.

## Reproductibilité (ENF06) : sur une même machine

Recherche exacte et départage par `passage_id` : deux exécutions sur une même machine donnent les mêmes résultats de recherche. En revanche, les vecteurs peuvent différer très légèrement d'une machine ou d'une version de torch à l'autre, à cause des calculs en virgule flottante et des instructions processeur utilisées. Deux scores très proches peuvent alors s'inverser. **ENF06 se vérifie donc sur une même machine.** Les versions des bibliothèques sont enregistrées dans les métadonnées de chaque index pour pouvoir comparer.

## Tests

| Fichier | Ce qu'il vérifie |
|---|---|
| `test_index.py` | Index exact, ex æquo, EF09 (rechargement identique, 5 000 × 1 024 en moins de 30 s, dossier accentué), D12 (chaque option change l'empreinte et force la reconstruction ; révision, passages), D11 (troncature), normalisation, métadonnées, index périmé refusé, EF11, préfixe de requête, résolution de la révision |
| `test_cli.py` | `rag ingest` construit puis recharge sans recharger le modèle ; `rag ask` ; index périmé refusé ; modèle inaccessible |
| `test_embeddings_reels.py` (`slow`) | Avec le vrai multilingual-e5-small : dimension 384, révision, **vecteurs de norme 1**, **4 questions plus proches du bon passage que d'un autre** (dont une en français face à des passages anglais), pas de préfixe doublé, tokens spéciaux comptés |

Les tests `slow` sont exclus par défaut, car ils téléchargent le modèle (environ 470 Mo au premier lancement). On les lance avec `python -m pytest -m slow`.

**Vérification des tests D12.** Après avoir écrit les tests, j'ai retiré volontairement le préfixe de requête de l'empreinte. Deux tests ont échoué : celui de l'empreinte et celui de la reconstruction. Ils détectent donc bien l'oubli qu'ils doivent empêcher.

## Exigences et tests

| Exigence | Critère | Test |
|---|---|---|
| EF08 | Changer de modèle = modifier la configuration | `test_index.py::test_d12_changer_une_seule_option_force_la_reconstruction[modele]` |
| EF09 | Index rechargé sans recalcul, en moins de 30 s | `test_index.py::test_ef09_index_recharge_donne_les_memes_resultats`, `test_ef09_rechargement_rapide_a_pleine_capacite`, `test_cli.py::test_ingest_construit_puis_recharge_l_index` |
| EF11 | k paramétrable, un score par passage | `test_index.py::test_ef11_k_parametrable_et_score_par_passage` |
| D8 | Le vrai modèle calcule des vecteurs pertinents | `test_embeddings_reels.py` (`slow`) |

## Mesure de temps : bge-m3 à 1 024 tokens (D4, ENF02)

À lancer sur ton ordinateur, sans GPU. Le premier lancement télécharge environ 2,3 Go :

```
rag ingest data/corpus/kubernetes --config configs/experiences/bge-m3-1024.yaml
```

La ligne `Index BAAI/bge-m3 : construit … passages en X s (Y passages/s)` donne la mesure ; `meta.json` la conserve. Le résultat, qu'il respecte ou non les 30 minutes d'ENF02, sera reporté ici et dans le rapport d'évaluation.

## Limites connues

- **Vecteurs** : ils peuvent différer très légèrement d'une machine à l'autre (voir Reproductibilité).
- **Pas de cache des vecteurs** : un passage modifié force le réencodage de tout l'index. C'est quelques minutes avec e5-small. Avec bge-m3 à 1 024 tokens, la mesure ci-dessus dira si un cache par passage devient nécessaire.
- **sentence-transformers et torch non exécutés dans mon environnement** : leur utilisation est vérifiée par la lecture du code source de la version figée et par les tests `slow`, lancés sur ton ordinateur.
