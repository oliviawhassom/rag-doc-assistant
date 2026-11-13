"""Génération de la réponse : prompt, LLM, langue (EF15–EF18, EF20, EF21)."""

from rag.generation.langue import Langue, detecter_langue, langue_question
from rag.generation.llm import (
    ContexteInsuffisant,
    Generateur,
    LLMIndisponible,
    ModeLocalViole,
    OllamaGenerateur,
    ReponseLLM,
    verifier_borne_contexte,
    verifier_mode_local,
)
from rag.generation.prompt import (
    PromptConstruit,
    VersionPrompt,
    charger_version,
    construire_prompt,
    neutraliser,
)

__all__ = [
    "ContexteInsuffisant",
    "Generateur",
    "LLMIndisponible",
    "Langue",
    "ModeLocalViole",
    "OllamaGenerateur",
    "PromptConstruit",
    "ReponseLLM",
    "VersionPrompt",
    "charger_version",
    "construire_prompt",
    "detecter_langue",
    "langue_question",
    "neutraliser",
    "verifier_borne_contexte",
    "verifier_mode_local",
]
