# Module 1 — Ingestion (EF01 à EF05)

Ce module transforme un dossier de fichiers en documents stockés, découpés en pages au texte nettoyé. Il ne fait encore ni découpage en passages, ni embeddings : c'est l'étape suivante.

## Le chemin d'un fichier

```
fichier
  │ validation.py   extension acceptée ? taille ≤ 50 Mo ? vrai PDF ?        → sinon rejet (EF03)
  │ pipeline.py     empreinte SHA-256 déjà connue ?                         → sinon doublon (EF04)
  │ extractors.py   texte page par page ; PDF chiffré ou corrompu ?         → rejet (EF03)
  │ cleaning.py     ligatures, Hugo/HTML, en-têtes répétés, césures, espaces (EF05)
  │ pipeline.py     reste-t-il du texte ?                                   → sinon rejet « PDF sans texte »
  ▼ store.py        pages/<id>.jsonl, puis une ligne dans documents.jsonl
```

Une erreur sur un fichier n'arrête pas l'import du dossier : le fichier est marqué rejeté avec sa cause, les autres continuent.

## Les fichiers, un par un

### `src/rag/config.py`
Lit `configs/default.yaml` et le valide avec Pydantic.
- **Clés inconnues refusées** (`extra="forbid"`). Une faute de frappe (`taille_max_mb` au lieu de `taille_max_mo`) fait échouer le chargement au lieu d'être ignorée en silence, ce qui fausserait une expérience sans qu'on le voie.
- **Configuration figée** (`frozen=True`) : elle ne change pas en cours d'exécution.
- Seule la section `ingestion` existe pour l'instant ; chaque module ajoutera la sienne.

### `src/rag/models.py`
`Document` reprend exactement les champs de la section 4.2. Deux précisions :
- **`nom_fichier` est le chemin relatif au dossier importé** (`en/concepts/workloads/pods/_index.md`), pas le seul nom de base : la documentation Kubernetes contient des dizaines de fichiers `_index.md`, et une citation `[_index.md, p. 1]` serait ambiguë.
- **`Page`** n'est pas dans le cahier des charges. C'est l'intermédiaire entre ingestion et découpage : le futur `Passage` (avec `page_debut`, `page_fin`) sera construit à partir des pages.

### `src/rag/ingestion/errors.py`
Une classe par cause de rejet, chacune avec un `code` stable (`pdf_mot_de_passe_requis`, `fichier_trop_lourd`…), un message en français et le **code HTTP que renverra l'API en V2** (413, 415, 422). La correspondance cause → code HTTP n'existe ainsi qu'à un endroit.

### `src/rag/ingestion/validation.py`
Contrôles bon marché, avant toute lecture du contenu : extension, taille, fichier vide, et pour un `.pdf` la présence de la signature `%PDF-`. Un texte renommé en `.pdf` est donc rejeté avant d'arriver à l'extracteur.

### `src/rag/ingestion/extractors.py`
Renvoie une liste de textes, **un par page, numérotation jamais décalée** : une page blanche reste une chaîne vide à sa place. Sinon toutes les citations après elle pointeraient une page trop tôt.
- **pypdf** plutôt que PyMuPDF (plus rapide, mais licence AGPL incompatible avec un dépôt MIT) ou pdfplumber (meilleure mise en page, nettement plus lent). Seule `_extract_pdf` serait à remplacer.
- **PDF chiffré** (décision D1) : rejeté seulement s'il exige un mot de passe d'ouverture. Un PDF qui n'a qu'un mot de passe propriétaire (interdiction d'imprimer ou de copier) s'ouvre avec un mot de passe utilisateur vide : il est accepté, et `extract_document` renvoie un avertissement que le pipeline journalise et que `rag ingest` affiche. `extract_pages` reste disponible quand seules les pages comptent.
- **TXT / Markdown** : UTF-8 (avec ou sans BOM), puis Windows-1252 pour les vieux fichiers. Un fichier qui contient des octets nuls est un binaire déguisé : rejeté.
- **Pages en TXT / Markdown** : un fichier = une page, sauf s'il contient des sauts de page `\f`.

### `src/rag/ingestion/cleaning.py`
Cinq étapes, chacune une fonction testable seule ; le nombre de pages ne change jamais.
1. **Caractères** : ligatures (`ﬁ` → `fi`), espaces insécables, tirets conditionnels. NFC plutôt que NFKC, qui transformerait `m²` en `m2`.
2. **Markdown Hugo** (désactivable) : le front matter devient un titre `# Pods` (le découpage par titres d'EF07 s'en servira), les ancres de fin de titre `{#...}` sont retirées (les titres deviennent des citations, décision D3), les shortcodes deviennent du texte (`{{< glossary_tooltip text="containers" >}}` → `containers`, `feature-state` → `(Feature state: stable since Kubernetes v1.29)`), liens → texte seul, HTML → texte. **Les blocs de code ne sont pas touchés**, et seules les balises HTML connues sont retirées : `<pod-name>` ou `<none>` restent.
3. **En-têtes et pieds de page** (PDF) : on regarde les 2 premières et 2 dernières lignes non vides de chaque page ; une ligne qui y revient sur au moins 50 % des pages qui ont du texte est retirée, ainsi que les numéros de page isolés. Les chiffres sont neutralisés pour que « Page 3 of 14 » et « Page 4 of 14 » soient reconnus comme la même ligne.
4. **Césures** (PDF uniquement) : `infor-\nmation` → `information`, mais pas devant une majuscule (`Jean-\nPierre`) ni un chiffre (`ISO-\n9001`). Pas en Markdown, où un tiret de fin de ligne est voulu.
5. **Espaces** : espaces multiples réduits, au plus une ligne vide, indentation de début de ligne conservée (listes et code Markdown).

### `src/rag/ingestion/store.py`
Fichiers JSONL plutôt que SQLite : quelques centaines de documents, lisibles avec un éditeur et faciles à comparer entre deux exécutions.
- **`document_id` dérivé de l'empreinte** (`doc_` + 12 premiers caractères du SHA-256) plutôt qu'un compteur : un même fichier a le même identifiant sur toutes les machines et après réindexation. Le jeu de test, qui référence des `document_id`, reste valide (ENF06).
- **Ordre d'écriture** : pages d'abord (renommage atomique), registre ensuite. Un arrêt brutal entre les deux ne laisse jamais le registre pointer vers des pages manquantes.

### `src/rag/ingestion/pipeline.py`
Enchaîne les étapes. L'empreinte est calculée **avant** l'extraction : un doublon est écarté sans payer l'extraction. Le contrôle « PDF sans couche texte » est fait **après** le nettoyage : un scan dont le seul texte est un en-tête ajouté par le logiciel de numérisation est bien rejeté. Seuil : moins de 20 caractères par page en moyenne.
`source_url` et `licence` viennent de `manifest.jsonl`, écrit par le script de téléchargement. Sans manifeste, l'URL est le chemin local et la licence « non renseignée », avec un avertissement.

### `src/rag/cli.py`
`rag ingest <chemin> --config <fichier>`. argparse plutôt que Typer : pas de dépendance pour quatre sous-commandes. Codes de sortie : 0 tout importé, 1 au moins un rejet, 2 erreur d'utilisation (configuration invalide, chemin introuvable) — utile dans un script ou en CI.

### `scripts/download_corpus.py`
Récupère par `git sparse-checkout` uniquement `content/en/docs/concepts/` (corpus en anglais, décision D2) du dépôt `kubernetes/website`, **à un commit figé** (branche `release-1.35`) : le corpus est identique partout. Écrit le manifeste avec l'URL GitHub exacte de chaque fichier à ce commit.

### `tests/`
Les PDF de test sont générés à la volée avec reportlab : aucun binaire dans le dépôt, et chaque page porte un marqueur unique (`MARQUEUR-P007`) qui permet de vérifier qu'elle est rangée sous le bon numéro.

## Exigences et tests

| Exigence | Critère du cahier des charges | Test |
|---|---|---|
| EF01 | Trois formats importés sans erreur | `test_pipeline.py::test_ef01_trois_formats_importes_sans_erreur` |
| EF02 | 20 pages tirées au hasard ont le bon numéro | `test_extractors.py::test_ef02_page_correcte_sur_20_pages_tirees_au_hasard` |
| EF03 | Trop lourd, mauvais format, sans texte ou exigeant un mot de passe d'ouverture : rejetés avec la cause. PDF chiffré lisible sans mot de passe : accepté, avertissement journalisé | `test_pipeline.py::test_ef03_chaque_rejet_donne_sa_cause`, `test_ef03_pdf_mot_de_passe_proprietaire_seul_accepte_avec_avertissement`, `test_ef03_pdf_mot_de_passe_ouverture_rejete` + `test_validation.py`, `test_extractors.py` |
| EF04 | Même empreinte → non réindexé | `test_pipeline.py::test_ef04_doublon_non_reindexe`, `test_ef04_meme_contenu_sous_un_autre_nom` |
| EF05 | Aucun en-tête répété sur 20 échantillons | `test_cleaning.py::test_ef05_aucun_en_tete_repete_sur_20_pages_tirees_au_hasard` |

EF02 et EF05 parlent de « passages » ; ils n'existent pas encore, donc les tests portent sur les pages. Ils seront rejoués sur les passages au module suivant, où EF02 vérifiera aussi la section pour les fichiers Markdown.

70 tests, couverture 96 % de `src/`, ruff et mypy strict sans erreur.

## Sur le vrai corpus

`rag ingest` sur les docs Kubernetes `concepts` en anglais : 166 fichiers importés en moins d'une seconde, aucun rejet, ~259 000 mots après nettoyage. Après nettoyage, plus aucun shortcode Hugo, lien Markdown ni balise HTML hors des blocs de code (vérifié par un balayage de toutes les pages ; ce balayage a fait apparaître deux défauts, corrigés et couverts par des tests : les shortcodes fermants écrits `{{</ note >}}` et une fin de lien sans crochet ouvrant présente dans la source). Un second balayage, dédié aux titres, a trouvé 549 titres abîmés (`## ## What's next`, ancres `{#...}`) ; corrigés, les 2 015 titres sont maintenant propres.

## Limites connues

- **Ligne de bord qui ne varie que par un nombre** : « Chapitre 3 — Installation » en haut de chaque page est retirée comme en-tête courant (voulu), mais une vraie ligne de contenu qui aurait cette forme sur la moitié des pages le serait aussi.
- **Césure devant une minuscule** : `state-\nof-the-art` perd son premier tiret.
- **Shortcodes sans contenu disponible** : `code_sample` (fichier YAML externe), `glossary_definition` (définition stockée ailleurs) et les schémas Mermaid sont retirés. Les exemples YAML référencés ne sont donc pas interrogeables.
- **Pages d'index de section** : 3 fichiers ne contiennent qu'un titre ou une phrase de renvoi. Ils sont importés ; le découpage décidera de les ignorer.
- **Ordre de lecture PDF** : sur une mise en page en colonnes, pypdf peut mélanger les colonnes.

## Décisions

Les trois questions posées à la livraison du module ont été tranchées le 8 novembre 2026. Elles sont consignées, avec leurs conséquences pour les modules suivants, dans [decisions.md](decisions.md) :

- **D1** — PDF chiffré : rejeté seulement s'il exige un mot de passe d'ouverture (appliquée ici).
- **D2** — Corpus en anglais, questions en anglais et en français (corpus appliqué ici ; embeddings multilingues et Recall@5 par langue aux modules 3 et 5).
- **D3** — Markdown : une page par fichier, citation [fichier, section] (titres nettoyés ici ; identifiant de section à fixer avant le module 2).

Taille du corpus : `concepts` en anglais fait ~259 000 mots après nettoyage, soit ~520 pages équivalentes, dans la fourchette 200–1 000 du cahier des charges.
