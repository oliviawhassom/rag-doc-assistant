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

*8 novembre 2026 · EF02, EF16, section 4.2 · statut : **ingestion et découpage appliqués** (modules 1 et 2) ; citations et jeu de test aux modules 4 et 5*

**Décision.** Un fichier Markdown est une seule page (`page` = 1). Comme certains fichiers Kubernetes sont longs, citer le fichier seul serait trop vague pour mesurer la précision des citations : on cite toujours **[fichier, section]**. Le jeu de test suit le même format : `sources_reference` contient (`document_id`, titre de section). Un PDF garde [document, page].

**Conséquence pour l'ingestion (faite).** Les titres doivent sortir propres, puisqu'ils deviennent des citations. Le contrôle du corpus a trouvé et fait corriger deux défauts qui touchaient 549 titres : les `## ## What's next` (shortcode `heading` doublé) et les ancres Hugo laissées en fin de titre (`## Historical context {#going-back-in-time}`). Les 2 015 titres du corpus sont maintenant propres, ce qui est vérifié par un balayage de toutes les pages et par `test_cleaning.py::test_titre_hugo_dans_une_ligne_de_titre_non_double` et `test_ancres_de_titre_retirees`.

**Identifiant de section (tranché, appliqué au module 2).** Le titre seul ne suffit pas : dans 9 fichiers, 14 titres apparaissent plusieurs fois (« Criteria » deux fois dans `pod-qos.md`, « Examples » dans `object-management.md`…). `titre_section` est donc le **chemin complet des titres** :

- titres de niveau 2 et plus, séparateur fixe ` > ` : `Imperative commands > Examples` ;
- titres nettoyés de leur mise en forme (code, gras, italique, liens, échappements, `#` de fermeture, espaces multiples), pour que deux personnes qui écrivent le même chemin obtiennent exactement la même chaîne ; un `>` présent dans un titre devient `›` ;
- le chemin complet est toujours stocké ; à l'affichage, une citation trop longue est raccourcie aux deux derniers niveaux (`… > Imperative object configuration > Examples`, fonction `chemin_affiche`).

**Précision ajoutée au module 2 : l'introduction a pour chemin `(introduction)`.** Le texte placé avant le premier titre de niveau 2 n'a pas de titre de section. Lui donner le titre du document créait une ambiguïté réelle : `persistent-volumes.md` s'intitule « Persistent Volumes » et contient une section `## Persistent Volumes`. Ma vérification précédente, qui concluait à « aucune ambiguïté », ignorait ce texte d'introduction : elle était fausse. Le test sur le corpus l'a révélé.

**Garde-fou.** `tests/test_corpus.py::test_aucun_chemin_de_section_ambigu_dans_le_corpus` échoue si un chemin désigne plusieurs sections d'un même fichier ; il reste valable si le corpus change. Le découpage signale aussi tout chemin ambigu, dans le journal et à l'écran.

## D4 — Tokenizer XLM-RoBERTa ; modèles multilingual-e5-small et bge-m3

*8 novembre 2026 · EF06, EF08, section 7.4 · statut : **tokenizer appliqué** (module 2) ; modèles au module 3*

**Décision.** Un token est celui du tokenizer XLM-RoBERTa, que partagent les deux modèles retenus : `intfloat/multilingual-e5-small` (petit, limite 512 tokens) et `BAAI/bge-m3` (grand, limite 8 192 tokens). Une taille de passage désigne ainsi exactement le même texte pour les deux modèles.

**Conséquences pour les expériences.** e5 attend les préfixes `query: ` et `passage: ` ; avec les tokens spéciaux, ils entrent dans sa limite de 512. **La comparaison des deux modèles se fait à 256 tokens** ; l'expérience sur la taille des passages (256, 512, 1 024) se fait avec bge-m3.

**Appliqué au module 2.** Les tailles sont comptées avec le tokenizer de bge-m3 (`decoupage.tokenizer`). Le comptage désactive la troncature intégrée au tokenizer, qui aurait compté trop court une longue section. Le texte compté est le contenu seul ; préfixes et chemin de section s'y ajoutent à l'encodage.

**À appliquer au module 3.**

| Point | Détail |
|---|---|
| Préfixes e5 | `query: ` pour les questions, `passage: ` pour les passages |
| Temps sur CPU | Mesurer le temps d'indexation avec bge-m3 à 1 024 tokens ; s'il dépasse les 30 minutes d'ENF02, le documenter plutôt que le cacher |
| Révisions | Hash du commit Hugging Face de chaque modèle dans la configuration et dans les rapports |

**Révision du tokenizer.** Elle est relevée à chaque découpage et écrite dans `store/passages/<config>/meta.json`. Tant que `decoupage.tokenizer.revision` vaut `main`, un avertissement donne le hash à recopier dans la configuration.

## D5 — Un passage ne franchit jamais une frontière de section

*8 novembre 2026 · EF06, EF07 · statut : **appliquée** (module 2)*

**Décision.** Avec la citation [fichier, section] (D3), un passage qui chevaucherait deux sections serait en partie cité sous la mauvaise section. Les deux stratégies découpent donc à l'intérieur de chaque section :

- **taille fixe** : fenêtres de N tokens avec chevauchement ; une dernière fenêtre qui n'apporterait que moins de 10 % de N tokens nouveaux est absorbée par la précédente, si celle-ci reste sous N + 10 % ;
- **par titres** : une section qui tient dans N tokens forme un seul passage ; une section plus longue est coupée entre deux paragraphes, et **jamais au milieu d'un bloc de code** ; un bloc (de code ou de prose) plus long que N passe en fenêtres fixes.

**Lecture d'EF06 (« taille respectée à 10 % près »).** Aucun passage ne dépasse N + 10 %. Avec la taille fixe, tout passage qui ne termine pas sa section atteint au moins N − 10 %. Avec le découpage par titres, seule la borne haute s'applique : les coupes tombent entre deux paragraphes, donc un passage peut être nettement plus court que N.

**Contexte des passages.** Le chemin de section précède le texte encodé (`contexte_section`, activé par défaut, identique dans toutes les expériences). Il n'est jamais affiché dans les réponses, qui citent le texte seul.

**Portée de l'expérience « stratégie de découpage ».** Les deux stratégies donnent exactement les mêmes passages pour une section qui tient dans N tokens : elles ne diffèrent que sur les sections plus longues. Leur proportion est calculée à chaque découpage (`proportion_sections_plus_longues` dans `meta.json`, et affichée par `rag ingest`) et devra figurer dans le rapport d'évaluation : sinon, une absence de différence entre stratégies serait mal interprétée.

## D6 — PDF et découpage « par titres » : rabattement sur la taille fixe

*8 novembre 2026 · EF07 · statut : **appliquée** (module 2)*

**Décision.** Aucun titre n'est extrait des PDF : la stratégie « par titres » s'y rabat sur la taille fixe, avec un avertissement journalisé et le document listé dans `documents_rabattus_taille_fixe`. Le corpus étant en Markdown, cela ne concerne que des PDF ajoutés plus tard.

**Limite connue, à reprendre si des PDF entrent dans le corpus.** Il serait possible de retrouver les titres d'un PDF, de deux façons : la taille de police, que pypdf peut donner par une fonction appelée pendant l'extraction ; les signets du PDF, qui donnent souvent la table des matières.

## D7 — Sections « What's next » exclues du découpage

*8 novembre 2026 · EF07 · statut : **appliquée** (module 2)*

**Décision.** Une section « What's next » n'est qu'une liste de liens vers d'autres pages : elle ressemble à beaucoup de questions sans répondre à aucune. Elle est exclue du découpage, avec ses sous-sections. La liste des titres exclus est réglable (`decoupage.sections_exclues`) ; la comparaison ignore la casse et les variantes d'apostrophe.

**Traçabilité.** Le nombre de sections exclues, par titre, est journalisé à chaque découpage, affiché par `rag ingest` et écrit dans `meta.json`. Au commit figé du corpus : 122 sections exclues.
