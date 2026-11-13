# Prompts versionnés (EF21, décision D16)

Chaque version est un dossier qui contient deux fichiers :

- `systeme.txt` : les instructions données au LLM ;
- `gabarit.txt` : le message qui contient les passages et la question, avec les champs `{passages}`, `{question}` et `{langue_reponse}`.

**Une version publiée ne se modifie jamais.** Pour changer un prompt, on crée une nouvelle version (`p2`, `p3`…) et on la sélectionne avec `generation.version_prompt` dans la configuration. Chaque réponse journalisée enregistre la version et l'empreinte SHA-256 des deux fichiers : une modification accidentelle d'une version existante se voit immédiatement.

Les instructions sont rédigées en anglais : les petits modèles les suivent plus fidèlement. La langue de la réponse n'est pas laissée au modèle : le code détecte la langue de la question et le gabarit l'indique explicitement (`{langue_reponse}`).

| Version | Changement |
|---|---|
| p1 | Version initiale |
