# Données : sources et licences

Le corpus n'est pas stocké dans ce dépôt. `scripts/download_corpus.py` le télécharge depuis sa source officielle, toujours au même commit, et écrit `data/corpus/kubernetes/manifest.jsonl` (URL d'origine et licence de chaque fichier). L'ingestion lit ce manifeste pour remplir les champs `source_url` et `licence` de chaque document.

| Corpus | Source | Version figée | Licence | Contenu retenu |
|---|---|---|---|---|
| Documentation Kubernetes | [github.com/kubernetes/website](https://github.com/kubernetes/website) | commit `8817d358864b` (branche `release-1.35`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `content/en/docs/concepts/` : 166 fichiers, ~259 000 mots après nettoyage, soit ~520 pages équivalentes. Anglais seul (décision D2) |

## Attribution (exigée par CC BY 4.0)

Contenu : documentation Kubernetes, © The Kubernetes Authors, licence CC BY 4.0. Le texte est modifié par le système (nettoyage des balises Hugo, découpage en passages) ; chaque réponse renvoie à l'URL du fichier d'origine.

## Notion de « page » et citations

Les fichiers sont en Markdown et n'ont pas de pages : un fichier est une page (`page` = 1). Comme certains fichiers sont longs, une citation désigne toujours le fichier **et la section** : `[en/concepts/workloads/pods/pod-qos.md, section]`. Le jeu de test suit le même format (décision D3, `docs/decisions.md`).
