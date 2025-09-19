"""Database initialisation utilities for the SQLAlchemy portal backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .sqlalchemy import Base, create_engine_from_url


def initialise_database(database_url: str, *, echo: bool = False) -> None:
    """Create or upgrade the database schema required by the portal."""

    engine = create_engine_from_url(database_url, echo=echo)
    Base.metadata.create_all(engine)


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a JSON object")
    return data


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initialise the HydroSIS portal database schema and create supporting indexes."
        )
    )
    parser.add_argument(
        "--database-url",
        dest="database_url",
        help="SQLAlchemy database URL. Overrides values from the configuration file.",
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="Optional path to a JSON configuration file containing a 'database_url' entry.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy engine echo for debugging.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    database_url = args.database_url
    if database_url is None and args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file '{config_path}' does not exist"
            )
        config = _load_config(config_path)
        config_url = config.get("database_url")
        if config_url and not isinstance(config_url, str):
            raise ValueError("database_url in configuration must be a string")
        database_url = config_url

    if not database_url:
        parser.error(
            "A database URL must be provided via --database-url or a configuration file."
        )

    initialise_database(database_url, echo=args.echo)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
