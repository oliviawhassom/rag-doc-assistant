"""Appel au LLM (décisions D13, D18 ; ENF04, ENF11, ENF14).

``Generateur`` est l'interface ; ``OllamaGenerateur`` l'implémentation locale,
par défaut. Le fournisseur externe (« api ») sera branché au module
d'évaluation, où il servira de juge ; pour l'instant il est refusé clairement.

Garde-fous :

- **Mode local (ENF11).** Avec ``mode_local``, l'adresse d'Ollama doit être
  cette machine (localhost, 127.0.0.1, ::1) ; tout autre hôte est refusé avant
  le moindre envoi, et le fournisseur externe aussi. Cela ne suffit pas : un
  Ollama local connecté à un compte ollama.com relaie les modèles « cloud »
  (étiquettes en « -cloud ») vers les serveurs d'Ollama. Un modèle dont
  l'étiquette contient « cloud », ou qu'Ollama décrit comme distant
  (champs ``remote_host`` / ``remote_model``), est donc refusé lui aussi, avant
  tout envoi de passages.
- **Fenêtre de contexte (D13).** Ollama tronque en silence un prompt plus long
  que ``num_ctx``, et il n'offre pas d'endpoint pour compter les tokens. Avant
  l'appel, on vérifie une borne garantie : un token couvre au moins un octet,
  donc ``octets du prompt + marge du gabarit + num_predict ≤ num_ctx`` exclut
  toute troncature. Si la borne n'est pas respectée, l'appel est refusé. Après
  l'appel, le nombre exact de tokens du prompt, compté par le LLM avec son
  propre tokenizer (``prompt_eval_count``), est relevé ; s'il approche
  ``num_ctx``, la réponse est rejetée.
- **Empreinte du modèle.** Un même nom (« gemma3:4b ») peut désigner des poids
  différents dans le temps : l'empreinte (digest) des poids est lue dans Ollama
  et jointe à chaque réponse.
- **Pannes (ENF14, D18).** Délai dépassé, Ollama arrêté, modèle absent, réponse
  vide : une erreur claire (``LLMIndisponible``), jamais une réponse inventée.

HTTP : bibliothèque standard (urllib), pour n'ajouter aucune dépendance.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from rag.config import GenerationConfig
from rag.generation.prompt import PromptConstruit

HOTES_LOCAUX = frozenset({"localhost", "127.0.0.1", "::1"})
# Au-delà de cette part de num_ctx, on soupçonne une troncature.
SEUIL_ALERTE_CONTEXTE = 0.95


class LLMIndisponible(RuntimeError):
    """Le LLM n'a pas produit de réponse exploitable (panne, délai, modèle absent)."""


class ContexteInsuffisant(RuntimeError):
    """Le prompt risque de dépasser la fenêtre de contexte : appel refusé."""


class ModeLocalViole(RuntimeError):
    """Configuration qui enverrait des données hors de la machine en mode local."""


@dataclass(frozen=True)
class ReponseLLM:
    texte: str
    modele: str
    empreinte_modele: str
    tokens_prompt: int
    tokens_generes: int
    duree_s: float
    # Durées internes d'Ollama, en secondes : chargement, lecture du prompt, génération.
    durees: dict[str, float] = field(default_factory=dict)


class Generateur(Protocol):
    modele: str

    def generer(self, prompt: PromptConstruit) -> ReponseLLM: ...


def est_modele_cloud(nom: str) -> bool:
    """Étiquette de modèle « cloud » d'Ollama (« gpt-oss:120b-cloud », « x:cloud »)."""
    etiquette = nom.split(":", 1)[1] if ":" in nom else ""
    return "cloud" in etiquette.lower()


def verifier_mode_local(cfg: GenerationConfig) -> None:
    if not cfg.mode_local:
        return
    if est_modele_cloud(cfg.modele):
        raise ModeLocalViole(
            f"mode local : « {cfg.modele} » est un modèle cloud d'Ollama, exécuté sur ses "
            "serveurs ; les passages quitteraient cette machine"
        )
    if cfg.fournisseur != "ollama":
        raise ModeLocalViole(
            f"mode local : le fournisseur « {cfg.fournisseur} » enverrait les passages "
            "à un service externe"
        )
    hote = urlparse(cfg.url).hostname or ""
    if hote not in HOTES_LOCAUX:
        raise ModeLocalViole(
            f"mode local : {cfg.url} n'est pas cette machine (attendu : localhost)"
        )


def verifier_borne_contexte(prompt: PromptConstruit, cfg: GenerationConfig) -> int:
    """Refuse un prompt qui pourrait dépasser num_ctx ; renvoie la borne en tokens."""
    borne = prompt.octets + cfg.marge_gabarit + cfg.num_predict
    if borne > cfg.num_ctx:
        raise ContexteInsuffisant(
            f"prompt de {prompt.octets} octets : avec la marge du gabarit "
            f"({cfg.marge_gabarit}) et la réponse ({cfg.num_predict}), il pourrait dépasser "
            f"num_ctx = {cfg.num_ctx} tokens et serait tronqué en silence. "
            "Augmenter generation.num_ctx ou réduire k ou la taille des passages."
        )
    return borne


class OllamaGenerateur:
    def __init__(self, cfg: GenerationConfig) -> None:
        verifier_mode_local(cfg)
        if cfg.fournisseur != "ollama":
            raise LLMIndisponible(
                f"fournisseur « {cfg.fournisseur} » non branché (prévu au module d'évaluation)"
            )
        self.cfg = cfg
        self.modele = cfg.modele
        self._url = cfg.url.rstrip("/")
        self._empreinte: str | None = None

    # --- HTTP ---------------------------------------------------------------------------

    def _requete(self, chemin: str, corps: dict[str, Any] | None = None) -> dict[str, Any]:
        donnees = json.dumps(corps).encode("utf-8") if corps is not None else None
        requete = urllib.request.Request(
            self._url + chemin,
            data=donnees,
            headers={"Content-Type": "application/json"},
            method="POST" if donnees is not None else "GET",
        )
        try:
            with urllib.request.urlopen(requete, timeout=self.cfg.delai_s) as reponse:
                resultat: dict[str, Any] = json.loads(reponse.read().decode("utf-8"))
                return resultat
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMIndisponible(f"Ollama a répondu {exc.code} : {detail}") from exc
        except TimeoutError as exc:
            raise LLMIndisponible(
                f"délai dépassé : pas de réponse d'Ollama en {self.cfg.delai_s:g} s"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LLMIndisponible(
                    f"délai dépassé : pas de réponse d'Ollama en {self.cfg.delai_s:g} s"
                ) from exc
            raise LLMIndisponible(
                f"Ollama injoignable à {self._url} ({exc.reason}). "
                "Est-il lancé ? (application Ollama, ou « ollama serve »)"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMIndisponible(f"réponse d'Ollama illisible : {exc}") from exc

    # --- Modèle -------------------------------------------------------------------------

    def empreinte_modele(self) -> str:
        """Empreinte (digest) des poids du modèle configuré, lue dans Ollama."""
        if self._empreinte is None:
            modeles = self._requete("/api/tags").get("models", [])
            noms = {self.modele, f"{self.modele}:latest"}
            for m in modeles:
                if m.get("name") in noms or m.get("model") in noms:
                    if self.cfg.mode_local and (m.get("remote_host") or m.get("remote_model")):
                        raise ModeLocalViole(
                            f"mode local : Ollama décrit « {self.modele} » comme un modèle "
                            f"distant ({m.get('remote_host') or m.get('remote_model')}) ; "
                            "les passages quitteraient cette machine"
                        )
                    self._empreinte = str(m.get("digest", ""))
                    break
            else:
                disponibles = ", ".join(sorted(str(m.get("name")) for m in modeles)) or "aucun"
                raise LLMIndisponible(
                    f"modèle « {self.modele} » absent d'Ollama (disponibles : {disponibles}). "
                    f"Le télécharger avec : ollama pull {self.modele}"
                )
        return self._empreinte

    def generer(self, prompt: PromptConstruit) -> ReponseLLM:
        verifier_borne_contexte(prompt, self.cfg)
        empreinte = self.empreinte_modele()
        corps: dict[str, Any] = {
            "model": self.modele,
            "messages": [
                {"role": "system", "content": prompt.systeme},
                {"role": "user", "content": prompt.message},
            ],
            "stream": False,
            "options": self.cfg.options_ollama(),
        }
        if self.cfg.reflexion is not None:
            corps["think"] = self.cfg.reflexion

        debut = time.perf_counter()
        reponse = self._requete("/api/chat", corps)
        duree = time.perf_counter() - debut

        texte = str(reponse.get("message", {}).get("content", "")).strip()
        if not texte:
            raise LLMIndisponible("le LLM a renvoyé une réponse vide")
        tokens_prompt = int(reponse.get("prompt_eval_count", 0))
        tokens_generes = int(reponse.get("eval_count", 0))
        if tokens_prompt + tokens_generes >= SEUIL_ALERTE_CONTEXTE * self.cfg.num_ctx:
            raise LLMIndisponible(
                f"contexte probablement saturé : {tokens_prompt} tokens de prompt + "
                f"{tokens_generes} générés pour num_ctx = {self.cfg.num_ctx} ; "
                "réponse rejetée (troncature possible)"
            )
        durees = {
            nom: round(int(reponse[cle]) / 1e9, 3)
            for nom, cle in (
                ("chargement", "load_duration"),
                ("lecture_prompt", "prompt_eval_duration"),
                ("generation", "eval_duration"),
            )
            if cle in reponse
        }
        return ReponseLLM(
            texte, self.modele, empreinte, tokens_prompt, tokens_generes, duree, durees
        )

    def modeles_charges(self) -> list[dict[str, Any]]:
        """Modèles en mémoire dans Ollama (/api/ps), avec la part placée sur le GPU."""
        modeles: list[dict[str, Any]] = self._requete("/api/ps").get("models", [])
        return modeles
