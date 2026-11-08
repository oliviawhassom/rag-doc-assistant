# Journal des décisions

Chaque décision indique ce qui a été tranché, pourquoi, et quels modules elle contraint. Une décision qui touche un module encore à écrire reste ici jusqu'à ce que ce module l'applique et la teste.

## D1 — PDF chiffré : rejeté seulement s'il exige un mot de passe d'ouverture

*8 novembre 2026 · EF03 · statut : **appliquée** (module 1)*

**Décision.** EF03 sert à rejeter ce qu'on ne peut pas lire. Un PDF qui n'a qu'un mot de passe propriétaire (restrictions d'impression ou de copie) s'ouvre et son texte s'extrait : il est accepté, et un avertissement est journalisé. Seuls les PDF qui exigent un mot de passe d'ouverture sont rejetés (code `pdf_mot_de_passe_requis`, HTTP 422).

**Mise en œuvre.** `extractors.py` tente d'ouvrir le PDF avec un mot de passe utilisateur vide. Si l'ouverture réussit, l'avertissement est renvoyé dans l'extraction ; le pipeline le journalise (niveau WARNING) et la commande `rag ingest` l'affiche.

**Tests.** `test_pipeline.py::test_ef03_pdf_mot_de_passe_proprietaire_seul_accepte_avec_avertissement` et `test_ef03_pdf_mot_de_passe_ouverture_rejete`.

## D2 — Corpus en anglais, questions en anglais et en français

*8 novembre 2026 · EF20, section 4.1, section 7 · statut : **corpus appliqué** ; le reste est à appliquer aux modules 3 et 5*

**Décision.** Le corpus est la documentation Kubernetes en anglais seule. Les questions sont posées en anglais et en français ; une question en français reçoit une réponse en français, avec des citations du texte anglais.

**Pourquoi.** La traduction française ne couvre que 59 fichiers sur 166 et duplique l'information anglaise : une même réponse aurait eu deux sources valides, ce qui rend `sources_reference` ambigu.

**Conditions.**

| Condition | Module | Raison |
|---|---|---|
| Le modèle d'embeddings est **multilingue** | 3 (embeddings) | Sinon une question en français retrouve mal les passages anglais, et l'échec vient du modèle, pas de la recherche. |
| Environ **20 % des questions en français**, avec un champ `langue` (`fr` / `en`) | 5 (jeu de test) | Pour mesurer l'écart entre les langues. |
| **Recall@5 donné séparément pour chaque langue** | 5 (évaluation) | Critère d'acceptation d'EF20. |
| La réponse est rédigée dans la langue de la question, les citations restent en anglais | 4 (génération) | EF20. |

**Déjà fait.** `scripts/download_corpus.py` ne télécharge plus que l'anglais (166 fichiers).

**À reporter dans le cahier des charges.** La nouvelle version reprend la décision (section 4.1, EF20) mais pas encore les deux premières conditions : le champ `langue` du modèle « Question de test » (section 4.2) et l'exigence d'un modèle d'embeddings multilingue (section 9.1, ligne « Modèles »).

## D3 — Markdown : une page par fichier, la section fait partie de la citation

*8 novembre 2026 · EF02, EF16, section 4.2 · statut : **ingestion appliquée** ; le reste est à appliquer aux modules 2, 4 et 5*

**Décision.** Un fichier Markdown est une seule page (`page` = 1). Comme certains fichiers Kubernetes sont longs, citer le fichier seul serait trop vague pour mesurer la précision des citations : on cite toujours **[fichier, section]**. Le jeu de test suit le même format : `sources_reference` contient (`document_id`, titre de section). Un PDF garde [document, page].

**Conséquence pour l'ingestion (faite).** Les titres doivent sortir propres, puisqu'ils deviennent des citations. Le contrôle du corpus a trouvé et fait corriger deux défauts qui touchaient 549 titres : les `## ## What's next` (shortcode `heading` doublé) et les ancres Hugo laissées en fin de titre (`## Historical context {#going-back-in-time}`). Les 2 015 titres du corpus sont maintenant propres, ce qui est vérifié par un balayage de toutes les pages et par `test_cleaning.py::test_titre_hugo_dans_une_ligne_de_titre_non_double` et `test_ancres_de_titre_retirees`.

**Point à trancher avant le module 2 : quel identifiant de section ?** Le titre seul ne suffit pas. Dans 9 fichiers, 14 titres apparaissent plusieurs fois : « Criteria » deux fois dans `pod-qos.md`, « Examples » et « Trade-offs » dans `object-management.md`, « Access Modes » dans `persistent-volumes.md`… Le couple (document_id, « Criteria ») ne désigne alors pas une section unique. Avec le **chemin complet des titres** (`Imperative commands > Examples`), aucune section n'est ambiguë dans les 166 fichiers (vérifié). Proposition : `titre_section` = chemin complet à partir des titres de niveau 2, séparé par « > ». C'est cette chaîne qui figurerait dans les citations et dans `sources_reference`.

**Autre point pour le module 2.** 118 fichiers finissent par une section « What's next » qui n'est qu'une liste de liens vers d'autres pages. L'indexer crée des passages qui ressemblent à beaucoup de questions sans y répondre. Proposition : l'exclure du découpage, par une liste de titres configurable.
