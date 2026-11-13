# Assistant RAG pour documentation technique

Répond en langage naturel à des questions sur un corpus de documentation technique, cite le document et la page de chaque information, et s'abstient quand le corpus ne contient pas la réponse. La qualité est mesurée de façon reproductible.

> **État : V1 en construction, module par module.** Ingestion (EF01–EF05), découpage (EF06–EF07), embeddings, index et recherche (EF08, EF09, EF11) livrés ; génération en cours (4a : client LLM et mesure des modèles). Voir `docs/` : [ingestion](docs/01-ingestion.md), [découpage](docs/02-decoupage.md), [embeddings et index](docs/03-embeddings-index.md), [génération](docs/04-generation.md), et le [journal des décisions](docs/decisions.md).

| Module | Exigences | État |
|---|---|---|
| Ingestion : import, extraction par page, nettoyage, doublons | EF01–EF05 | ✅ |
| Découpage en passages : sections, taille fixe ou par titres | EF06, EF07 | ✅ |
| Embeddings, index FAISS et recherche | EF08, EF09, EF11 | ✅ |
| Génération, citations, abstention | EF15–EF18, EF20, EF21 | en cours (4a livré) |
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

# 2. Importer, nettoyer, découper et indexer les documents
rag ingest data/corpus/kubernetes --config configs/default.yaml

# 3. Rechercher les passages les plus proches d'une question
rag ask "How do I create a Pod?" --config configs/default.yaml
```

`python -m rag` est équivalent à `rag` et fonctionne même si le dossier des commandes Python n'est pas dans le PATH.

La commande importe les documents, puis découpe tout le corpus en passages selon la section `decoupage` de la configuration. Le résultat est écrit dans `store/` :
- `documents.jsonl` et `pages/` : les documents et le texte nettoyé de chaque page ;
- `passages/<configuration>/` : les passages et un `meta.json` avec les statistiques du découpage.

Relancer la commande ne réimporte que ce qui a changé : les fichiers déjà connus sont comptés comme doublons, et ceux nettoyés par une ancienne version du nettoyage sont retraités.

**Premier lancement** : il faut un accès à huggingface.co, une fois. Sont téléchargés le tokenizer de `BAAI/bge-m3` (environ 17 Mo) et le modèle `intfloat/multilingual-e5-small` (environ 470 Mo), aux révisions figées dans `configs/default.yaml`. Ensuite, le cache local suffit. Sur un ordinateur portable sans GPU, l'indexation du corpus prend environ 2 minutes.

**Expériences** : `configs/experiences/` contient des configurations complètes qui ne diffèrent de la configuration par défaut que par les paramètres étudiés. Chaque configuration a ses propres passages et son propre index dans `store/`.

## Génération (en cours)

Le LLM tourne en local avec [Ollama](https://ollama.com). Une fois Ollama installé, mesurer les modèles candidats sur ce poste :

```bash
ollama pull gemma3:4b
ollama pull qwen3.5:4b
ollama pull granite4.2:3b
python scripts/mesurer_llm.py --modeles gemma3:4b qwen3.5:4b granite4.2:3b
```

Le rapport (latences, débit, langue, abstentions, citations, réponses complètes) est écrit dans `rapports/mesure-llm/`.

## Vérifications

```bash
python -m pytest       # tests + couverture (sans les tests « slow »)
python -m pytest -m slow   # tests avec les vrais modèles (e5, et le LLM si Ollama est lancé)
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
  index/                    encodeur, index FAISS, indexation, recherche
  generation/               prompts, client Ollama, langue, citations, abstention
prompts/                    prompts versionnés (p1, …)
rapports/                   mesures et rapports produits par les scripts
tests/                      un test par critère d'acceptation
docs/                       explications des choix, module par module
```

## Licence

Code sous licence MIT. Le corpus a sa propre licence : voir [data/README.md](data/README.md).
