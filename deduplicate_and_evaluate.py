#!/usr/bin/env python3
"""Trajno uklanja duplikate iz sačuvanih API rezultata i ponavlja evaluaciju.

Skripta ne poziva NVIDIA/Azure modele. Za svaki agregirani_rezultati.json
zadržava prvo pojavljivanje svakog teksta, isto kao data.load_dataset(), zatim
pokreće 04_dekoderski_modeli.py nad očišćenim agregatima.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise argparse.ArgumentTypeError("Navedite barem jedan model.")
    return models


def discover_aggregates(root: Path) -> list[tuple[str, Path]]:
    """Nalazi root agregat ili agregate u neposrednim prompt poddirektorijumima."""
    direct = root / "agregirani_rezultati.json"
    if direct.exists():
        return [(root.name, direct)]

    found = [
        (path.parent.name, path)
        for path in sorted(root.glob("*/agregirani_rezultati.json"))
        if path.parent.name != "analiza"
    ]
    if not found:
        raise ValueError(
            f"Nema agregirani_rezultati.json u {root} niti u neposrednim poddirektorijumima."
        )
    return found


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Fajl ne postoji: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Neispravan JSON {path}: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} mora biti neprazna JSON lista.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Svaki zapis u {path} mora biti JSON objekat.")
    return payload


def deduplicate_records(
    records: list[dict[str, Any]],
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Zadržava prvi zapis po tekst.strip(), kao pandas drop_duplicates."""
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, record in enumerate(records, start=1):
        text = record.get("tekst")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source_path}: zapis {position} nema neprazno polje 'tekst'.")
        groups[text.strip()].append((position, record))

    cleaned: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    prediction_conflicts: list[dict[str, Any]] = []
    gold_conflicts: list[dict[str, Any]] = []

    for normalized_text, occurrences in groups.items():
        gold_labels = {
            str(record.get("sentiment", "")).strip().lower()
            for _, record in occurrences
        }
        if len(gold_labels) != 1:
            gold_conflicts.append({
                "tekst": normalized_text,
                "pozicije": [position for position, _ in occurrences],
                "oznake": sorted(gold_labels),
                "zadrzana_oznaka": str(occurrences[0][1].get("sentiment", "")).strip().lower(),
                "napomena": "Zadržana je oznaka prvog pojavljivanja, kao kod pandas drop_duplicates.",
            })

        first_position, first_record = occurrences[0]
        kept = dict(first_record)
        kept["tekst"] = normalized_text
        cleaned.append(kept)

        if len(occurrences) == 1:
            continue

        first_predictions = first_record.get("model_predictions")
        differing_positions = [
            position
            for position, record in occurrences[1:]
            if record.get("model_predictions") != first_predictions
        ]
        if differing_positions:
            prediction_conflicts.append({
                "tekst": normalized_text,
                "zadrzana_pozicija": first_position,
                "razlicite_pozicije": differing_positions,
                "napomena": "Zadržana je predikcija prvog pojavljivanja, kao kod pandas drop_duplicates.",
            })

        kept_id = first_record.get("id")
        for position, record in occurrences[1:]:
            removed.append({
                "uklonjena_pozicija": position,
                "uklonjeni_id": record.get("id"),
                "zadrzani_id": kept_id,
                "tekst": normalized_text,
            })

    report = {
        "izvor": str(source_path),
        "pravilo": "tekst.strip(); zadržava se prvo pojavljivanje",
        "broj_pre": len(records),
        "broj_posle": len(cleaned),
        "uklonjeno": len(records) - len(cleaned),
        "broj_grupa_sa_duplikatima": sum(len(items) > 1 for items in groups.values()),
        "broj_grupa_sa_konfliktnim_gold_oznakama": len(gold_conflicts),
        "broj_grupa_sa_razlicitim_predikcijama": len(prediction_conflicts),
        "konflikti_gold_oznaka": gold_conflicts,
        "konflikti_predikcija": prediction_conflicts,
        "uklonjeni_zapisi": removed,
    }
    return cleaned, report


def validate_models(records: list[dict[str, Any]], models: list[str], path: Path) -> None:
    for position, record in enumerate(records, start=1):
        predictions = record.get("model_predictions")
        if not isinstance(predictions, dict):
            raise ValueError(f"{path}: zapis {position} nema model_predictions objekat.")
        for model in models:
            result = predictions.get(model)
            if not isinstance(result, dict):
                raise ValueError(f"{path}: zapis {position} nema rezultat modela {model!r}.")
            prediction = result.get("sentiment")
            if prediction not in VALID_SENTIMENTS and not result.get("error"):
                raise ValueError(
                    f"{path}: zapis {position}, model {model!r}, nema validnu predikciju ni grešku."
                )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Uklanja duplikate iz sačuvanih NVIDIA/Azure agregata i ponavlja metrike bez API poziva."
    )
    parser.add_argument("input_root", type=Path, help="Run direktorijum sa agregatom ili prompt poddirektorijumima.")
    parser.add_argument("output_root", type=Path, help="Novi direktorijum za očišćene rezultate i analizu.")
    parser.add_argument(
        "--models", type=parse_models, required=True,
        help="Tačni model ključevi odvojeni zarezom, npr. gpt-5-mini.",
    )
    parser.add_argument(
        "--analysis-script", type=Path,
        default=Path(__file__).resolve().parent / "claudeNLP1" / "04_dekoderski_modeli.py",
        help="Putanja do 04_dekoderski_modeli.py.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Samo napravi očišćene JSON fajlove, bez računanja metrika/grafika.",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        parser.error("Ulazni i izlazni direktorijum moraju biti različiti; izvor se nikad ne prepisuje.")

    try:
        aggregates = discover_aggregates(input_root)
        cleaned_inputs: list[tuple[str, Path]] = []
        combined_report: dict[str, Any] = {
            "ulazni_direktorijum": str(input_root),
            "izlazni_direktorijum": str(output_root),
            "modeli": args.models,
            "eksperimenti": {},
        }

        for experiment_name, source_path in aggregates:
            records = load_records(source_path)
            validate_models(records, args.models, source_path)
            cleaned, report = deduplicate_records(records, source_path)

            target_dir = output_root / experiment_name if len(aggregates) > 1 else output_root
            target_path = target_dir / "agregirani_rezultati.json"
            write_json(target_path, cleaned)
            write_json(target_dir / "izvestaj_duplikata.json", report)

            prompt_config = source_path.parent / "prompt_config.json"
            if prompt_config.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prompt_config, target_dir / "prompt_config.json")

            cleaned_inputs.append((experiment_name, target_path))
            combined_report["eksperimenti"][experiment_name] = report
            print(
                f"{experiment_name}: {report['broj_pre']} -> {report['broj_posle']} "
                f"(uklonjeno {report['uklonjeno']}, "
                f"konflikti gold oznaka {report['broj_grupa_sa_konfliktnim_gold_oznakama']}, "
                f"konflikti predikcija {report['broj_grupa_sa_razlicitim_predikcijama']})"
            )

        write_json(output_root / "izvestaj_duplikata_sve.json", combined_report)

        if args.skip_analysis:
            print(f"Očišćeni rezultati: {output_root}")
            return 0

        analysis_script = args.analysis_script.resolve()
        if not analysis_script.exists():
            raise ValueError(f"Skripta za analizu ne postoji: {analysis_script}")

        command = [
            sys.executable,
            str(analysis_script),
            "--models", ",".join(args.models),
            "--output-name", "deduplicated_precomputed",
            "--output-dir", str(output_root / "analiza"),
        ]
        for experiment_name, target_path in cleaned_inputs:
            command.extend(["--result", f"{experiment_name}={target_path}"])

        print("Pokrećem ponovnu evaluaciju bez API poziva...")
        completed = subprocess.run(command, cwd=analysis_script.parent, check=False)
        if completed.returncode != 0:
            raise ValueError(f"Analiza je završena kodom {completed.returncode}.")
        print(f"Očišćeni rezultati i nova analiza: {output_root}")
        return 0
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
