# Module 2 — Découpage en passages (EF06, EF07)

Ce module transforme les pages nettoyées en passages : les extraits que la recherche retrouvera et que les réponses citeront. Il ne calcule pas encore d'embeddings : c'est le module 3.

Les choix de ce module viennent des décisions D3 à D7, détaillées dans [decisions.md](decisions.md).

## Le chemin d'un document

```
pages nettoyées (module 1)
  │ decoupage.py   Markdown ? → sections.py : sections et chemins de titres (D3)
  │                              sections exclues comptées puis écartées (D7)
  │                PDF / TXT ? → une seule « section » sans titre ; « par titres » → taille fixe (D6)
  │ splitter.py    chaque section découpée selon la stratégie, sans jamais en sortir (D5)
  │                tailles comptées en tokens XLM-RoBERTa (tokenizer.py, D4)
  ▼ store.py       store/passages/<config_decoupage>/<document_id>.jsonl + meta.json
```

`rag ingest` enchaîne l'import (module 1) et le découpage de tout le corpus, conformément à la section 6.1 (« importe et indexe »).

## Les fichiers, un par un

### `configs/default.yaml` et `src/rag/config.py` — section `decoupage`

| Réglage | Défaut | Rôle |
|---|---|---|
| `strategie` | `taille_fixe` | ou `titres` (EF07) |
| `taille` | 256 | en tokens ; 256 est la taille de comparaison des modèles (D4) |
| `chevauchement` | 32 | en tokens, taille fixe seulement ; refusé s'il dépasse la moitié de la taille |
| `tokenizer.modele` | `BAAI/bge-m3` | tokenizer qui définit « un token » |
| `tokenizer.revision` | `main` | hash du commit Hugging Face, à figer (voir plus bas) |
| `sections_exclues` | `["What's next"]` | D7 |
| `contexte_section` | `true` | chemin de section en tête du texte encodé (D5) |

**Identifiant de configuration.** `config_decoupage` vaut par exemple `taille_fixe-256-32-1a2b3c4d` : stratégie, taille et chevauchement pour la lisibilité, puis une empreinte de toute la section. Deux réglages qui diffèrent sur un seul point (une exclusion, le contexte, la révision du tokenizer) n'ont jamais le même identifiant, donc jamais les mêmes fichiers de passages.

### `src/rag/models.py` — `Passage`

Les champs de la section 4.2 du cahier des charges, sans ajout.
- **`titre_section`** contient le chemin complet (`Pod lifecycle > Pod phase`) pour un Markdown. Il vaut `None` pour un PDF ou un texte brut, où la citation se fait par la page.
- **`page_debut` et `page_fin`** valent 1 pour un Markdown (D3). Pour un PDF, un passage peut couvrir deux pages.
- **`passage_id`** vaut `<document_id>_c0000`, numéroté dans l'ordre du document. Il n'est unique qu'au sein d'une configuration, d'où le champ `config_decoupage`.

### `src/rag/chunking/tokenizer.py`

Il charge le `tokenizer.json` de bge-m3 avec la bibliothèque `tokenizers`, via le cache Hugging Face : téléchargé au premier lancement (17 Mo environ), relu en local ensuite.
- **On ne s'en sert que pour des positions.** Pour chaque token, le tokenizer donne son début et sa fin dans le texte. Un passage est une **tranche exacte du texte d'origine**, coupée entre deux tokens. On ne reconstruit jamais du texte à partir des tokens, ce qui altérerait espaces et caractères.
- **Troncature désactivée.** Le `tokenizer.json` de bge-m3 tronque par défaut. Sans cette précaution, une longue section aurait été comptée trop courte, sans erreur.
- **Révision relevée.** Le hash du commit réellement téléchargé est lu dans le chemin du cache (`snapshots/<hash>/`). Il est écrit dans `meta.json`, et un avertissement le donne tant que la configuration dit `main`.

### `src/rag/chunking/sections.py`

Il découpe un Markdown en sections et calcule leur chemin (D3).
- **Titres nettoyés** (`nettoyer_titre`) : code, gras, italique, liens, échappements, `#` de fermeture, ancres et espaces multiples. ``The `Pod` API`` et `The Pod API ##` donnent la même chaîne. Un `\*` échappé n'est pas pris pour de l'italique, et `max_surge` reste intact.
- **Titres cherchés hors des blocs de code**, quelle que soit leur indentation. Dans la documentation Kubernetes, des blocs de code sont indentés sous des éléments de liste. Avec la limite stricte de CommonMark (3 espaces), leurs commentaires `# …` auraient été pris pour des titres.
- **Introduction** : le texte placé avant le premier titre de niveau 2 a pour chemin `(introduction)`. Voir D3 pour la raison.
- **`chemins_ambigus`** liste les chemins portés par plusieurs sections d'un même fichier. Le découpage le signale, et `tests/test_corpus.py` en fait un test sur tout le corpus.
- **`chemin_affiche`** raccourcit un chemin long aux deux derniers niveaux, pour l'affichage uniquement.

### `src/rag/chunking/splitter.py`

Les deux stratégies, appliquées à une section à la fois (D5). Elles renvoient des intervalles de caractères.
- **`fenetres`** (taille fixe) : fenêtres de N tokens, chevauchement de C tokens, donc un pas de N − C. Une dernière fenêtre qui n'apporterait que quelques tokens nouveaux est absorbée par la précédente, si celle-ci reste sous N + 10 %. Cela évite les passages de 3 tokens.
- **`blocs`** : paragraphes séparés par une ligne vide. Un bloc de code forme un seul bloc, même s'il contient des lignes vides, et même s'il n'est jamais refermé.
- **`par_blocs`** (par titres) : les blocs consécutifs sont regroupés tant que le groupe tient dans N tokens. Un bloc plus long que N, code compris, passe en fenêtres fixes. Il n'y a pas de chevauchement entre groupes, puisque les coupes tombent entre deux paragraphes.

### `src/rag/chunking/decoupage.py`

Il découpe tout le corpus et produit les statistiques.
- **Redécoupage complet à chaque appel** (environ une seconde pour 166 fichiers). Les passages reflètent toujours le code et la configuration actuels : pas de passages périmés à gérer.
- **Statistiques** (`meta.json`, journal, écran) : sections, sections plus longues que N et leur proportion (D5), sections exclues par titre (D7), documents rabattus sur la taille fixe (D6), chemins ambigus, distribution des tailles, passages au-dessus de la tolérance.
- **`texte_a_encoder`** : le texte que le module 3 encodera, avec le chemin de section en tête si `contexte_section` est activé. Le préfixe `passage: ` de e5 sera ajouté par le module 3.

### `src/rag/chunking/store.py`

Un dossier par configuration, `store/passages/<config_decoupage>/`. Les expériences (256, 512 ou 1 024 tokens ; taille fixe ou par titres) coexistent sans s'écraser. L'écriture se fait dans un dossier temporaire substitué d'un coup : un arrêt en cours ne laisse jamais un mélange d'ancien et de nouveau découpage. La séquence de renommage est compatible avec Windows, qui refuse de renommer par-dessus un dossier existant.

### `src/rag/cli.py`

`rag ingest` importe puis découpe, et affiche une ligne de résumé : passages, sections, proportion de sections longues, sections exclues. Il renvoie le code 2 si le tokenizer est inaccessible. Les fichiers déjà connus ne sont plus listés un par un (166 lignes à chaque relance) : seul leur total est affiché, et `-v` les détaille. Les fichiers « mis à jour » (version du nettoyage) restent listés.

### `tests/`

| Fichier | Ce qu'il vérifie |
|---|---|
| `test_sections.py` | Nettoyage des titres, chemins, introduction, blocs de code, ambiguïtés, exclusions, affichage |
| `test_splitter.py` | Fenêtres, chevauchement, absorption de la queue, tolérance d'EF06, blocs, regroupement, code jamais coupé |
| `test_decoupage.py` | Passages de bout en bout, EF02 et EF05 rejoués sur les passages, EF07, D6, statistiques, déterminisme |
| `test_corpus.py` | Sur le vrai corpus : aucun chemin ambigu, aucun titre avec résidu, exclusions présentes. Sauté si le corpus est absent |
| `test_tokenizer_reel.py` | Avec le vrai tokenizer XLM-RoBERTa : révision relevée, pas de troncature, tolérance d'EF06. Sauté si Hugging Face est injoignable |

Les autres tests utilisent un vrai objet `tokenizers.Tokenizer` au vocabulaire vide : un mot ou un signe de ponctuation = un token. C'est le même code que chez toi, avec des comptes vérifiables à la main.

## Exigences et tests

| Exigence | Critère | Test |
|---|---|---|
| EF06 | Taille respectée à 10 % près (lecture D5) | `test_splitter.py::test_ef06_tolerance_sur_un_long_texte`, `test_tokenizer_reel.py::test_ef06_tolerance_avec_le_vrai_tokenizer`, `test_ef06_sur_un_document_du_corpus` |
| EF06 | Taille et chevauchement paramétrables | `test_splitter.py::test_fenetres_taille_et_chevauchement_exacts` |
| EF07 | Stratégie choisie dans la configuration | `test_decoupage.py::test_ef07_strategie_choisie_par_la_configuration` |
| EF02 | Page (PDF) ou section (Markdown) correcte sur 20 passages | `test_decoupage.py::test_ef02_pages_des_passages_pdf`, `test_passages_markdown` |
| EF05 | Aucun en-tête répété dans 20 passages | `test_decoupage.py::test_ef05_aucun_en_tete_dans_20_passages` |
| D3 | Aucun chemin ambigu dans le corpus | `test_corpus.py::test_aucun_chemin_de_section_ambigu_dans_le_corpus` |
| D7 | Sections exclues journalisées | `test_decoupage.py::test_decouper_corpus_ecrit_passages_et_statistiques` |

## Sur le vrai corpus

Mesuré sur Windows avec le vrai tokenizer XLM-RoBERTa (bge-m3, révision `5617a9f61b028005a4858fdac845db406aefb181`), configuration par défaut (taille fixe, 256 tokens, chevauchement 32) :

| Mesure | Valeur |
|---|---|
| Documents | 166 |
| Sections avec du contenu | 1 833 |
| Sections plus longues que 256 tokens | 587, soit **32,0 %** |
| Sections « What's next » exclues | 122 |
| Passages | 2 657 |
| Chemins ambigus | 0 |

**Ce que signifient les 32,0 % (D5).** Pour 68 % des sections, les deux stratégies d'EF07 donnent exactement les mêmes passages, puisque ces sections tiennent dans 256 tokens. L'expérience « stratégie de découpage » ne peut montrer une différence que par les 32 % restants. Cette proportion diminuera quand la taille augmente : à 512 ou 1 024 tokens, moins de sections dépasseront la taille, et les deux stratégies se ressembleront encore davantage.

Le tokenizer factice des tests (un mot = un token) donnait environ 22 % : le vrai découpe plus finement qu'un mot par token.

Le découpage du corpus a fait trouver et corriger trois défauts du nettoyage du module 1 (version 3 du nettoyage, voir [01-ingestion.md](01-ingestion.md)), et une erreur dans ma vérification préliminaire des ambiguïtés (voir D3).

## Limites connues

- **Titres des PDF** (D6) : non extraits. La taille de police, que pypdf peut donner pendant l'extraction, ou les signets du PDF le permettraient. Cela n'a pas été fait, puisque le corpus est en Markdown.
- **Titres Markdown soulignés** (`Titre\n=====`) : non reconnus, seuls les titres `#` le sont. La documentation Kubernetes n'en contient pas.
- **Taille des passages « par titres »** : seule la borne haute d'EF06 s'applique (lecture D5). Un passage peut être bien plus court que N quand le paragraphe suivant ne tient pas.
- **Premier lancement** : il faut un accès à huggingface.co pour télécharger le tokenizer. Ensuite, le cache local suffit.
