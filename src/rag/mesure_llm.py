"""Mesure comparée des LLM candidats sur ce poste (décision D13).

Pour chaque modèle : latence de chaque réponse, débit de lecture du prompt et
de génération, part du modèle placée sur le GPU (doit valoir 0 : ENF04),
empreinte des poids, et réponses complètes à relire pour juger la qualité du
français. Tous les modèles reçoivent exactement les mêmes prompts (mêmes
passages retrouvés, même version de prompt) : seule la variable « modèle »
change.

Deux vérifications supplémentaires :

- **Déterminisme** : la première question est reposée à la fin. Avec une
  température de 0 et une graine fixe, la réponse doit être identique.
- **Compte des tokens du prompt** : Ollama peut réutiliser en cache le début
  d'un prompt déjà vu. Si ``prompt_eval_count`` diminue à la seconde lecture,
  il ne compte que les tokens recalculés, et ce point doit être pris en compte
  par la détection de saturation du contexte (D13).

Les questions ci-dessous servent uniquement à cette mesure : elles ne font
partie ni du jeu de calibration ni du jeu de test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Protocol

from rag.generation.analyse import contient_marqueur, extraire_citations
from rag.generation.langue import Langue, detecter_langue
from rag.generation.llm import LLMIndisponible, ReponseLLM
from rag.generation.prompt import PromptConstruit


@dataclass(frozen=True)
class QuestionMesure:
    identifiant: str
    langue: Langue
    texte: str
    hors_corpus: bool  # True : la bonne réponse est l'abstention


QUESTIONS_MESURE = (
    QuestionMesure("en-pod", "en", "How do I create a Pod?", False),
    QuestionMesure("fr-service", "fr", "Comment exposer une application avec un Service ?", False),
    QuestionMesure("en-secret", "en", "How can a Pod use the data stored in a Secret?", False),
    QuestionMesure("fr-daemonset", "fr", "À quoi sert un DaemonSet ?", False),
    QuestionMesure(
        "en-statefulset", "en", "When should I use a StatefulSet instead of a Deployment?", False
    ),
    QuestionMesure(
        "fr-memoire",
        "fr",
        "Que se passe-t-il quand un conteneur dépasse sa limite de mémoire ?",
        False,
    ),
    QuestionMesure("en-hors", "en", "What is the capital city of Australia?", True),
    QuestionMesure(
        "fr-hors",
        "fr",
        "Quel est le prix mensuel d'un cluster Kubernetes managé chez un fournisseur cloud ?",
        True,
    ),
)


class GenerateurMesurable(Protocol):
    modele: str

    def generer(self, prompt: PromptConstruit) -> ReponseLLM: ...
    def empreinte_modele(self) -> str: ...
    def modeles_charges(self) -> list[dict[str, Any]]: ...


def _analyser(question: QuestionMesure, prompt: PromptConstruit, r: ReponseLLM) -> dict[str, Any]:
    citations = extraire_citations(r.texte, len(prompt.passages))
    return {
        "question": question.identifiant,
        "langue_attendue": question.langue,
        "langue_reponse": detecter_langue(r.texte),
        "hors_corpus": question.hors_corpus,
        "abstention": contient_marqueur(r.texte),
        "citations_valides": list(citations.valides),
        "citations_invalides": list(citations.invalides),
        "duree_s": round(r.duree_s, 3),
        "tokens_prompt": r.tokens_prompt,
        "tokens_generes": r.tokens_generes,
        "durees": r.durees,
        "texte": r.texte,
    }


def mesurer_modele(
    generateur: GenerateurMesurable,
    prompts: list[tuple[QuestionMesure, PromptConstruit]],
) -> dict[str, Any]:
    """Mesure un modèle ; renvoie un dictionnaire sérialisable en JSON."""
    resultat: dict[str, Any] = {"modele": generateur.modele}
    try:
        resultat["empreinte"] = generateur.empreinte_modele()
        # Premier appel : charge le modèle en mémoire. Mesuré à part, il n'entre
        # pas dans les latences (un service garde le modèle chargé).
        q0, p0 = prompts[0]
        chauffe = generateur.generer(p0)
    except LLMIndisponible as exc:
        resultat["erreur"] = str(exc)
        return resultat
    resultat["chauffe_s"] = round(chauffe.duree_s, 3)
    resultat["chargement_s"] = chauffe.durees.get("chargement")

    charges = generateur.modeles_charges()
    infos = next((m for m in charges if m.get("name", "").startswith(generateur.modele)), {})
    resultat["taille_memoire"] = infos.get("size")
    resultat["taille_gpu"] = infos.get("size_vram")

    reponses = []
    for question, prompt in prompts:
        try:
            reponses.append(_analyser(question, prompt, generateur.generer(prompt)))
        except LLMIndisponible as exc:
            reponses.append({"question": question.identifiant, "erreur": str(exc)})
    resultat["reponses"] = reponses

    # Déterminisme et cache : la première question, reposée.
    try:
        reprise = generateur.generer(p0)
        premiere = next(r for r in reponses if r["question"] == q0.identifiant)
        resultat["determinisme"] = {
            "texte_identique": reprise.texte == premiere.get("texte"),
            "tokens_prompt_premiere": premiere.get("tokens_prompt"),
            "tokens_prompt_reprise": reprise.tokens_prompt,
        }
    except (LLMIndisponible, StopIteration) as exc:
        resultat["determinisme"] = {"erreur": str(exc)}

    resultat["synthese"] = synthese(reponses)
    return resultat


def synthese(reponses: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in reponses if "erreur" not in r]
    durees = sorted(r["duree_s"] for r in ok)
    lecture = [
        (r["tokens_prompt"], r["durees"]["lecture_prompt"])
        for r in ok
        if r["durees"].get("lecture_prompt")
    ]
    generation = [
        (r["tokens_generes"], r["durees"]["generation"])
        for r in ok
        if r["durees"].get("generation")
    ]
    dans_corpus = [r for r in ok if not r["hors_corpus"]]
    hors_corpus = [r for r in ok if r["hors_corpus"]]
    return {
        "questions": len(reponses),
        "erreurs": len(reponses) - len(ok),
        "latence_mediane_s": round(statistics.median(durees), 2) if durees else None,
        "latence_max_s": round(durees[-1], 2) if durees else None,
        "lecture_prompt_tokens_s": (
            round(sum(t for t, _ in lecture) / sum(d for _, d in lecture), 1) if lecture else None
        ),
        "generation_tokens_s": (
            round(sum(t for t, _ in generation) / sum(d for _, d in generation), 1)
            if generation
            else None
        ),
        "tokens_prompt_moyen": round(statistics.mean(r["tokens_prompt"] for r in ok))
        if ok
        else None,
        "bonne_langue": sum(r["langue_reponse"] == r["langue_attendue"] for r in dans_corpus),
        # Réponse trop courte ou trop technique pour que la langue soit détectée :
        # à vérifier à la lecture, ce n'est pas forcément une erreur.
        "langue_indeterminee": sum(r["langue_reponse"] is None for r in dans_corpus),
        "reponses_dans_corpus": len(dans_corpus),
        "abstentions_hors_corpus": sum(r["abstention"] for r in hors_corpus),
        "questions_hors_corpus": len(hors_corpus),
        "abstentions_a_tort": sum(r["abstention"] for r in dans_corpus),
        "reponses_sans_citation": sum(
            not r["abstention"] and not r["citations_valides"] for r in dans_corpus
        ),
        "citations_invalides": sum(len(r["citations_invalides"]) for r in ok),
    }


def rapport_markdown(resultats: list[dict[str, Any]], contexte: dict[str, Any]) -> str:
    """Rapport lisible : tableau comparatif, puis toutes les réponses à relire."""
    lignes = [
        "# Mesure des LLM candidats",
        "",
        f"Date : {contexte['date']} · Prompt : {contexte['version_prompt']} · "
        f"num_ctx : {contexte['num_ctx']} · num_gpu : {contexte['num_gpu']} · "
        f"k : {contexte['k']} · Index : {contexte['index']}",
        "",
        "Latences : appel au LLM seul, modèle déjà chargé. « GPU » : mémoire du modèle "
        "placée sur le GPU (0 attendu, ENF04).",
        "",
        "| Modèle | GPU | Latence médiane | Latence max | Lecture prompt | Génération "
        "| Bonne langue | Abstentions hors corpus | Abstentions à tort | Sans citation "
        "| Citations invalides | Déterministe |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in resultats:
        if "erreur" in r:
            lignes.append(f"| {r['modele']} | erreur : {r['erreur']} |" + " |" * 10)
            continue
        s = r["synthese"]
        det = r.get("determinisme", {}).get("texte_identique")
        lignes.append(
            f"| {r['modele']} | {r.get('taille_gpu')} | {s['latence_mediane_s']} s "
            f"| {s['latence_max_s']} s | {s['lecture_prompt_tokens_s']} tok/s "
            f"| {s['generation_tokens_s']} tok/s "
            f"| {s['bonne_langue']}/{s['reponses_dans_corpus']}"
            + (f" ({s['langue_indeterminee']} indét.)" if s["langue_indeterminee"] else "")
            + " "
            f"| {s['abstentions_hors_corpus']}/{s['questions_hors_corpus']} "
            f"| {s['abstentions_a_tort']} | {s['reponses_sans_citation']} "
            f"| {s['citations_invalides']} | {'oui' if det else 'non'} |"
        )
    lignes += ["", "## Réponses", ""]
    for r in resultats:
        if "erreur" in r:
            continue
        lignes += [f"### {r['modele']}", "", f"Empreinte : `{r['empreinte']}`", ""]
        d = r.get("determinisme", {})
        lignes += [
            f"Tokens du prompt de la 1re question : {d.get('tokens_prompt_premiere')} "
            f"puis {d.get('tokens_prompt_reprise')} à la reprise (cache Ollama si différent).",
            "",
        ]
        for rep in r["reponses"]:
            lignes.append(f"**{rep['question']}**")
            if "erreur" in rep:
                lignes += [f"> Erreur : {rep['erreur']}", ""]
                continue
            lignes += [
                f"{rep['duree_s']} s · {rep['tokens_prompt']} tokens de prompt · "
                f"langue {rep['langue_reponse']} · citations {rep['citations_valides']}"
                + (
                    f" · **invalides {rep['citations_invalides']}**"
                    if rep["citations_invalides"]
                    else ""
                ),
                "",
                *(f"> {ligne}" for ligne in rep["texte"].splitlines()),
                "",
            ]
    return "\n".join(lignes) + "\n"
