"""Permet ``python -m rag ...``, équivalent à la commande ``rag``.

Utile quand le dossier des scripts de Python n'est pas dans le PATH
(cas fréquent sous Windows sans environnement virtuel).
"""

import sys

from rag.cli import main

sys.exit(main())
