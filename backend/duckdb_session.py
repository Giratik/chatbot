# backend/duckdb_session.py
#
# Stack simplifiée :
#   Lecture   → openpyxl (détection d'îlots existante conservée)
#   Stockage  → DuckDB in-memory par session (RAM uniquement, zéro disque)
#   Dialogue  → Schéma injecté directement dans le system prompt (zéro embeddings)
#   Graphes   → SQL → Polars → Plotly (côté frontend)
#
# Cycle de vie : ExcelSession vit le temps de la session HTTP.
# Destruction = libération mémoire immédiate, rien à nettoyer sur disque.

import io
import uuid
import threading
from pathlib import Path
from typing import Optional

import duckdb as _duckdb
import openpyxl
import polars as pl

import re

# ---------------------------------------------------------------------------
# Détection des îlots — logique conservée de l'ancien xlsx_parser.py
# ---------------------------------------------------------------------------

def _identifier_ilots(df: pl.DataFrame) -> list[dict]:
    """
    Identifie les blocs de données séparés par des lignes/colonnes vides.
    Conservé tel quel depuis l'ancien xlsx_parser.py.
    """
    if df.is_empty():
        return []

    masque = df.select([
        pl.all().is_not_null() & (pl.all().cast(pl.Utf8) != "")
    ])

    lignes_pleines = masque.select(pl.any_horizontal(pl.all())).to_series().to_list()

    blocs = []
    start_row = None

    for i, est_pleine in enumerate(lignes_pleines):
        if est_pleine and start_row is None:
            start_row = i
        elif not est_pleine and start_row is not None:
            df_temp = masque.slice(start_row, i - start_row)
            cols_with_data = [
                j for j in range(len(df_temp.columns))
                if df_temp[df_temp.columns[j]].any()
            ]
            if cols_with_data:
                start_col = cols_with_data[0]
                for idx in range(len(cols_with_data)):
                    curr_col = cols_with_data[idx]
                    next_col = cols_with_data[idx + 1] if idx + 1 < len(cols_with_data) else None
                    if next_col is not None and next_col - curr_col > 1:
                        blocs.append({"start_row": start_row, "end_row": i,
                                      "start_col": start_col, "end_col": curr_col + 1})
                        start_col = next_col
                if start_col is not None:
                    blocs.append({"start_row": start_row, "end_row": i,
                                  "start_col": start_col, "end_col": cols_with_data[-1] + 1})
            start_row = None

    # Dernier bloc non terminé par une ligne vide
    if start_row is not None:
        df_temp = masque.slice(start_row, len(masque) - start_row)
        cols_with_data = [
            j for j in range(len(df_temp.columns))
            if df_temp[df_temp.columns[j]].any()
        ]
        if cols_with_data:
            start_col = cols_with_data[0]
            for idx in range(len(cols_with_data)):
                curr_col = cols_with_data[idx]
                next_col = cols_with_data[idx + 1] if idx + 1 < len(cols_with_data) else None
                if next_col is not None and next_col - curr_col > 1:
                    blocs.append({"start_row": start_row, "end_row": len(masque),
                                  "start_col": start_col, "end_col": curr_col + 1})
                    start_col = next_col
            if start_col is not None:
                blocs.append({"start_row": start_row, "end_row": len(masque),
                              "start_col": start_col, "end_col": cols_with_data[-1] + 1})

    return blocs


def _lire_sheet_openpyxl(file_bytes: bytes, sheet_name: str) -> pl.DataFrame:
    """
    Lit une feuille Excel via openpyxl et retourne un DataFrame Polars brut
    avec toutes les valeurs converties en string (évite les conflits de types).
    """
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    ws = wb[sheet_name]

    rows_data = list(ws.iter_rows(values_only=True))
    if not rows_data:
        return pl.DataFrame()

    max_cols = max(len(row) for row in rows_data)
    normalized = [
        [str(v) if v is not None else None for v in row] + [None] * (max_cols - len(row))
        for row in rows_data
    ]

    try:
        return pl.DataFrame(
            normalized,
            schema=[f"col_{i}" for i in range(max_cols)],
            infer_schema_length=None,
        )
    except Exception:
        return pl.DataFrame(
            normalized,
            schema={f"col_{i}": pl.Utf8 for i in range(max_cols)},
        )


def find_tables_in_sheet(file_bytes: bytes, sheet_name: str) -> list[tuple[str, pl.DataFrame, str | None]]:
    df_raw = _lire_sheet_openpyxl(file_bytes, sheet_name)
    if df_raw.is_empty():
        return []

    ilots = _identifier_ilots(df_raw)
    tables = []

    for i, coord in enumerate(ilots):
        df_bloc = df_raw.slice(coord["start_row"], coord["end_row"] - coord["start_row"])
        cols = [df_raw.columns[j] for j in range(coord["start_col"], coord["end_col"])]
        df_bloc = df_bloc.select(cols)
        df_bloc = df_bloc.filter(~pl.all_horizontal(pl.all().is_null()))
        if df_bloc.is_empty():
            continue

        # Détection titre
        first_row = df_bloc.row(0)
        non_null_values = [(j, v) for j, v in enumerate(first_row) if v is not None and str(v).strip() != ""]

        if len(non_null_values) == 1:
            table_title = str(non_null_values[0][1]).strip()
            df_bloc = df_bloc.slice(1)
            if df_bloc.is_empty():
                continue
        else:
            table_title = None

        # Promotion header
        header_row = df_bloc.row(0)
        new_cols = [str(h).strip() if h and str(h).strip() else f"col_{j}" for j, h in enumerate(header_row)]
        
        # Déduplication colonnes
        seen: dict[str, int] = {}
        deduped = []
        for col in new_cols:
            if col in seen:
                seen[col] += 1
                deduped.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                deduped.append(col)

        df_final = df_bloc.slice(1)
        df_final.columns = deduped  # ← la ligne manquante

        # Nom SQL avec déduplication
        if table_title:
            base_name = re.sub(r'[^a-zA-Z0-9]', '_', table_title).lower().strip('_')
            table_name = base_name if base_name else f"tableau_{i + 1}"
        else:
            table_name = f"tableau_{i + 1}"

        original_name = table_name
        counter = 2
        while table_name in [t[0] for t in tables]:
            table_name = f"{original_name}_{counter}"
            counter += 1

        tables.append((table_name, df_final, table_title))

    return tables


# ---------------------------------------------------------------------------
# Session DuckDB — une instance par utilisateur
# ---------------------------------------------------------------------------

class ExcelSession:
    """
    Encapsule un DuckDB in-memory pour une session utilisateur.

    Cycle de vie :
      - Créée à l'upload du fichier Excel
      - Détruite explicitement via .close() ou par le garbage collector
      - Aucune écriture sur disque

    Thread-safety : DuckDB gère ses propres locks internes.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        # ":memory:" = base 100% RAM, détruite à la fermeture de la connexion
        self.con = _duckdb.connect(":memory:")
        self.tables: dict[str, dict] = {}  # nom_table → métadonnées
        print(f"✅ Session DuckDB créée [{self.session_id}]")

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def load_excel(self, file_bytes: bytes, sheet_name: str) -> list[str]:
        """
        Parse le fichier Excel, enregistre chaque îlot comme table DuckDB.
        Retourne la liste des noms de tables créées.
        """
        tables_found = find_tables_in_sheet(file_bytes, sheet_name)
        if not tables_found:
            raise ValueError(f"Aucun tableau trouvé dans la feuille '{sheet_name}'.")

        created = []
        for table_name, df, table_title in tables_found:
            self.con.register(table_name, df)
            self.tables[table_name] = {
                "title": table_title or table_name,
                "columns": df.columns,
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "n_rows": len(df),
                "n_cols": len(df.columns),
            }
            created.append(table_name)
            print(f"  📊 Table '{table_name}' chargée ({len(df)} lignes × {len(df.columns)} cols)")

        print(f"✅ {len(created)} table(s) disponible(s) dans la session [{self.session_id}]")
        return created

    # ------------------------------------------------------------------
    # Requêtes
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Exécute du SQL et retourne un DataFrame Polars."""
        return self.con.execute(sql).pl()

    def query_to_records(self, sql: str) -> list[dict]:
        """Exécute du SQL et retourne une liste de dicts (JSON-serializable)."""
        df = self.query(sql)
        return df.to_dicts()

    # ------------------------------------------------------------------
    # Génération du schéma pour le LLM
    # ------------------------------------------------------------------

    def build_schema_prompt(self) -> str:
        """
        Construit la description des tables à injecter dans le system prompt.
        Typiquement < 500 tokens pour un fichier Excel standard.

        Format :
          tableau_1 (120 lignes)
            - date : Utf8
            - montant : Float64
            - region : Utf8
        """
        if not self.tables:
            return "Aucune table disponible."

        lines = ["Tables disponibles (DuckDB) :\n"]
        for name, meta in self.tables.items():
            title_info = f" — \"{meta['title']}\"" if meta['title'] != name else ""
            lines.append(f"  {name}{title_info} ({meta['n_rows']} lignes)")
            for col, dtype in meta["dtypes"].items():
                lines.append(f"    - {col} : {dtype}")
            lines.append("")

        return "\n".join(lines)

    def get_tables_info(self) -> list[dict]:
        """Retourne les métadonnées de toutes les tables (pour l'API)."""
        return [
            {"name": name, **meta}
            for name, meta in self.tables.items()
        ]

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    def close(self):
        """Libère la connexion DuckDB et toute la RAM associée."""
        self.con.close()
        self.tables.clear()
        print(f"🧹 Session DuckDB fermée [{self.session_id}]")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registre global des sessions (thread-safe)
# ---------------------------------------------------------------------------

class SessionRegistry:
    """
    Registre thread-safe des sessions DuckDB actives.
    Une session par session_id utilisateur.
    """

    def __init__(self):
        self._sessions: dict[str, ExcelSession] = {}
        self._lock = threading.Lock()

    def create(self, session_id: Optional[str] = None) -> ExcelSession:
        """Crée une nouvelle session (remplace l'ancienne si même ID)."""
        with self._lock:
            sid = session_id or str(uuid.uuid4())[:8]
            # Fermer l'ancienne session si elle existe
            if sid in self._sessions:
                self._sessions[sid].close()
            session = ExcelSession(session_id=sid)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Optional[ExcelSession]:
        """Récupère une session existante, None si introuvable."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_raise(self, session_id: str) -> ExcelSession:
        """Récupère une session ou lève une ValueError explicite."""
        session = self.get(session_id)
        if session is None:
            raise ValueError(
                f"Session '{session_id}' introuvable. "
                "Veuillez d'abord uploader un fichier Excel."
            )
        return session

    def delete(self, session_id: str):
        """Ferme et supprime une session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].close()
                del self._sessions[session_id]

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Instance globale — importée par main.py
registry = SessionRegistry()