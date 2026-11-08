# Assistant RAG pour documentation technique

Répond en langage naturel à des questions sur un corpus de documentation technique, cite le document et la page de chaque information, et s'abstient quand le corpus ne contient pas la réponse. La qualité est mesurée de façon reproductible.

> **État : V1 en construction, module par module.** Ingestion (EF01–EF05) et découpage en passages (EF06–EF07) livrés. Voir [docs/01-ingestion.md](docs/01-ingestion.md), [docs/02-decoupage.md](docs/02-decoupage.md) et le [journal des décisions](docs/decisions.md).

| Module | Exigences | État |
|---|---|---|
| Ingestion : import, extraction par page, nettoyage, doublons | EF01–EF05 | ✅ |
| Découpage en passages : sections, taille fixe ou par titres | EF06, EF07 | ✅ |
| Embeddings et index | EF08, EF09 | à faire |
| Recherche | EF11 | à faire |
| Génération, citations, abstention | EF15–EF18, EF20, EF21 | à faire |
| Évaluation | EF22, EF23, EF26 | à faire |
| Journalisation | EF27, EF28 | à faire |

## Installation

Python 3.11 ou plus récent, et git (pour télécharger le corpus). Une commande par ligne : elles fonctionnent telles quelles sous Windows PowerShell, macOS et Linux.

```bash
python -m venv .venv
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

Entre la première et la deuxième ligne, activer l'environnement virtuel :

| Système | Commande d'activation |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Une fois activé, l'invite commence par `(.venv)`. Si PowerShell refuse d'exécuter le script d'activation (« l'exécution de scripts est désactivée »), autoriser les scripts locaux pour l'utilisateur courant puis réessayer :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Utilisation

```bash
# 1. Télécharger le corpus (documentation Kubernetes, CC BY 4.0, commit figé)
python scripts/download_corpus.py

# 2. Importer, nettoyer et découper les documents
rag ingest data/corpus/kubernetes --config configs/default.yaml
```

`python -m rag` est équivalent à `rag` et fonctionne même si le dossier des commandes Python n'est pas dans le PATH.

La commande importe les documents, puis découpe tout le corpus en passages selon la section `decoupage` de la configuration. Le résultat est écrit dans `store/` :
- `documents.jsonl` et `pages/` : les documents et le texte nettoyé de chaque page ;
- `passages/<configuration>/` : les passages et un `meta.json` avec les statistiques du découpage.

Relancer la commande ne réimporte que ce qui a changé : les fichiers déjà connus sont comptés comme doublons, et ceux nettoyés par une ancienne version du nettoyage sont retraités.

**Premier lancement** : le découpage télécharge le tokenizer de `BAAI/bge-m3` (environ 17 Mo, accès à huggingface.co nécessaire une fois). Un avertissement affiche alors la révision téléchargée : recopie-la dans `decoupage.tokenizer.revision` de `configs/default.yaml`, pour que le découpage soit reproductible.

## Vérifications

```bash
python -m pytest       # tests + couverture
ruff check .
ruff format --check .
mypy
```

## Structure

```
configs/default.yaml        tous les réglages (ENF15)
data/README.md              sources et licences du corpus
scripts/download_corpus.py  téléchargement reproductible du corpus
src/rag/
  config.py                 chargement et validation de la configuration
  models.py                 Document, Page, Passage
  cli.py                    commande « rag »
  ingestion/                validation, extraction, nettoyage, stockage
  chunking/                 sections, découpage, tokenizer, stockage des passages
tests/                      un test par critère d'acceptation
docs/                       explications des choix, module par module
```

## Licence

Code sous licence MIT. Le corpus a sa propre licence : voir [data/README.md](data/README.md).
