"""Mesure les LLM candidats sur ce poste, avec les mêmes prompts (décision D13).

Prérequis : Ollama lancé, les modèles téléchargés (ollama pull <modèle>), et
l'index construit (rag ingest).

    python scripts/mesurer_llm.py --modeles gemma3:4b qwen3.5:4b granite4.2:3b

Résultats dans rapports/mesure-llm/<date>/ : resultats.json (tout) et
rapport.md (tableau comparatif et réponses complètes à relire).

Les options de génération (num_ctx, num_gpu = 0, température, graine…) sont
celles de la configuration ; seul le modèle change d'une mesure à l'autre.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rag.chunking import PassageStore
from rag.config import load_config
from rag.generation import (
    LLMIndisponible,
    OllamaGenerateur,
    charger_version,
    construire_prompt,
    langue_question,
)
from rag.index import MoteurRecherche, charger_encodeur, charger_index_a_jour, resoudre_revision
from rag.mesure_llm import QUESTIONS_MESURE, mesurer_modele, rapport_markdown

RACINE = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=RACINE / "configs" / "default.yaml")
    parser.add_argument("--modeles", nargs="+", required=True, help="étiquettes Ollama")
    parser.add_argument("--sortie", type=Path, default=RACINE / "rapports" / "mesure-llm")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s : %(message)s")

    cfg = load_config(args.config)
    g = cfg.generation

    # 1. Mêmes passages pour tous les modèles : recherche faite une seule fois.
    passages = PassageStore(cfg.stockage.dossier, cfg.decoupage.identifiant).lire_tous()
    revision = resoudre_revision(cfg.embeddings.modele, cfg.embeddings.revision)
    index, meta_index = charger_index_a_jour(passages, cfg, revision)
    moteur = MoteurRecherche(
        index, passages, charger_encodeur(cfg.embeddings, revision), cfg.embeddings
    )
    version = charger_version(g.version_prompt)
    prompts = []
    for q in QUESTIONS_MESURE:
        langue, _ = langue_question(q.texte, g.langue_defaut)
        trouves = [r.passage for r in moteur.rechercher(q.texte, cfg.recherche.k)]
        prompts.append((q, construire_prompt(q.texte, trouves, langue, version)))
    print(f"{len(prompts)} questions, passages retrouvés avec {cfg.embeddings.modele}.\n")

    # 2. Chaque modèle, avec les options de la configuration.
    resultats = []
    for modele in args.modeles:
        print(f"Mesure de {modele}…", flush=True)
        resultat = None
        # Les modèles à « réflexion » (qwen3.5…) doivent l'avoir désactivée, sinon
        # la latence explose ; les autres refusent ce paramètre : on le retire alors.
        for reflexion in (False, None):
            generateur = OllamaGenerateur(
                g.model_copy(update={"modele": modele, "reflexion": reflexion})
            )
            resultat = mesurer_modele(generateur, prompts)
            if "think" not in resultat.get("erreur", "").lower():
                break
        assert resultat is not None
        resultat["reflexion_envoyee"] = reflexion
        resultats.append(resultat)
        if "erreur" in resultat:
            print(f"  erreur : {resultat['erreur']}")
        else:
            s = resultat["synthese"]
            print(
                f"  latence médiane {s['latence_mediane_s']} s, max {s['latence_max_s']} s · "
                f"GPU {resultat['taille_gpu']} · bonne langue {s['bonne_langue']}/"
                f"{s['reponses_dans_corpus']} · abstentions hors corpus "
                f"{s['abstentions_hors_corpus']}/{s['questions_hors_corpus']}"
            )

    # 3. Rapport.
    contexte = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "version_prompt": f"{version.version} (systeme {version.empreinte_systeme[:12]}, "
        f"gabarit {version.empreinte_gabarit[:12]})",
        "num_ctx": g.num_ctx,
        "num_gpu": g.num_gpu,
        "k": cfg.recherche.k,
        "index": f"{meta_index['modele']} {meta_index['revision'][:10]}",
        "options": g.options_ollama(),
    }
    dossier = args.sortie / datetime.now().strftime("%Y%m%d-%H%M%S")
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "resultats.json").write_text(
        json.dumps({"contexte": contexte, "resultats": resultats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dossier / "rapport.md").write_text(rapport_markdown(resultats, contexte), encoding="utf-8")
    print(f"\nRapport : {dossier / 'rapport.md'}")
    return 0 if all("erreur" not in r for r in resultats) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LLMIndisponible as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(2)
