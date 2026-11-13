"""Mesure comparée des LLM (D13) : logique testée avec un générateur factice."""

from __future__ import annotations

from typing import Any

from rag.generation import LLMIndisponible, ReponseLLM, charger_version, construire_prompt
from rag.generation.prompt import PromptConstruit
from rag.mesure_llm import QUESTIONS_MESURE, mesurer_modele, rapport_markdown, synthese
from rag.models import Passage

PASSAGE = Passage(
    passage_id="d_c0000",
    document_id="d",
    page_debut=1,
    page_fin=1,
    titre_section="S",
    texte="Create a Pod with kubectl apply.",
    nb_tokens=7,
    config_decoupage="c",
)


def _prompts() -> list[tuple[Any, PromptConstruit]]:
    v = charger_version("p1")
    return [(q, construire_prompt(q.texte, [PASSAGE], q.langue, v)) for q in QUESTIONS_MESURE]


class GenerateurFactice:
    """Répond en français ou en anglais selon la question ; s'abstient hors corpus."""

    def __init__(self, absent: bool = False, cache: bool = False) -> None:
        self.modele = "factice:4b"
        self._absent = absent
        self._cache = cache
        self._vus: set[str] = set()

    def empreinte_modele(self) -> str:
        if self._absent:
            raise LLMIndisponible("modèle « factice:4b » absent d'Ollama")
        return "sha256:abc"

    def modeles_charges(self) -> list[dict[str, Any]]:
        return [{"name": "factice:4b", "size": 4_000_000_000, "size_vram": 0}]

    def generer(self, prompt: PromptConstruit) -> ReponseLLM:
        message = prompt.message
        if "Australia" in message or "prix mensuel" in message:
            texte = "INTROUVABLE_DANS_LES_PASSAGES"
        elif "Answer in French" in message:
            texte = "Créez un Pod avec la commande kubectl apply [1]."
        else:
            texte = "Create a Pod with kubectl apply [1]."
        tokens = 500
        if self._cache and message in self._vus:
            tokens = 20  # Ollama ne compte que les tokens recalculés
        self._vus.add(message)
        durees = {"chargement": 0.0, "lecture_prompt": 2.0, "generation": 1.0}
        return ReponseLLM(texte, self.modele, "sha256:abc", tokens, 20, 3.0, durees)


def test_mesure_complete() -> None:
    r = mesurer_modele(GenerateurFactice(), _prompts())
    s = r["synthese"]
    assert r["empreinte"] == "sha256:abc"
    assert r["taille_gpu"] == 0
    assert s["questions"] == 8 and s["erreurs"] == 0
    assert s["bonne_langue"] == 6 and s["reponses_dans_corpus"] == 6
    assert s["abstentions_hors_corpus"] == 2 and s["questions_hors_corpus"] == 2
    assert s["abstentions_a_tort"] == 0
    assert s["reponses_sans_citation"] == 0
    assert s["latence_mediane_s"] == 3.0
    assert s["lecture_prompt_tokens_s"] == 250.0  # 500 tokens en 2 s
    assert s["generation_tokens_s"] == 20.0
    assert r["determinisme"]["texte_identique"] is True


def test_effet_du_cache_sur_le_compte_de_tokens() -> None:
    r = mesurer_modele(GenerateurFactice(cache=True), _prompts())
    d = r["determinisme"]
    # La 1re question a déjà été vue à la chauffe : le cache réduit déjà son compte.
    assert d["tokens_prompt_reprise"] < 500


def test_modele_absent_signale_sans_mesure() -> None:
    r = mesurer_modele(GenerateurFactice(absent=True), _prompts())
    assert "absent" in r["erreur"]
    assert "reponses" not in r


def test_synthese_compte_les_ecarts() -> None:
    reponses = [
        {
            "question": "a",
            "langue_attendue": "fr",
            "langue_reponse": "en",
            "hors_corpus": False,
            "abstention": True,
            "citations_valides": [],
            "citations_invalides": [9],
            "duree_s": 12.0,
            "tokens_prompt": 100,
            "tokens_generes": 10,
            "durees": {},
            "texte": "x",
        },
        {"question": "b", "erreur": "délai dépassé"},
    ]
    s = synthese(reponses)
    assert s["erreurs"] == 1
    assert s["bonne_langue"] == 0
    assert s["langue_indeterminee"] == 0
    assert s["abstentions_a_tort"] == 1
    assert s["citations_invalides"] == 1
    assert s["latence_max_s"] == 12.0
    assert s["lecture_prompt_tokens_s"] is None


def test_rapport_markdown() -> None:
    resultats = [
        mesurer_modele(GenerateurFactice(), _prompts()),
        mesurer_modele(GenerateurFactice(absent=True), _prompts()),
    ]
    contexte = {
        "date": "2026-11-09",
        "version_prompt": "p1",
        "num_ctx": 16384,
        "num_gpu": 0,
        "k": 5,
        "index": "e5",
    }
    md = rapport_markdown(resultats, contexte)
    attendu = "| factice:4b | 0 | 3.0 s | 3.0 s | 250.0 tok/s | 20.0 tok/s "
    assert attendu + "| 6/6 | 2/2 | 0 | 0 | 0 | oui |" in md
    assert "erreur : modèle « factice:4b » absent" in md
    assert "> Créez un Pod avec la commande kubectl apply [1]." in md
