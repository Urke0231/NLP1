#!/usr/bin/env python3
"""Klasifikacija finansijskog sentimenta kroz NVIDIA NIM modele.

Svaki model obrađuje ulazne zapise redom, uz zasebno ograničenje broja
zahteva. Tri modela se izvršavaju paralelno, pa spor ili neuspešan zahtev
jednog modela ne blokira preostala dva.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import requests


API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODELS = (
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
    # "moonshotai/kimi-k2.6",
    "mistralai/mistral-medium-3.5-128b",
    "deepseek-ai/deepseek-v4-flash"
)
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

SYSTEM_PROMPT = """Ti si strogi klasifikator finansijskog sentimenta za tekstove na srpskom jeziku.

Vrati ISKLJUČIVO jednu malu englesku reč: positive, negative ili neutral. Ne dodaj objašnjenje, interpunkciju ni JSON.

Pravila:
- positive: povoljan poslovni, finansijski ili ekonomski razvoj, npr. rast prihoda, dobiti, prodaje, izvoza, investicija, zaposlenosti, tržišnog učešća, kreditnog rejtinga ili poverenja potrošača; najava poboljšanja poslovanja, stabilizacije tržišta ili uspešne strategije.
- negative: nepovoljan poslovni, finansijski ili ekonomski razvoj, npr. pad prihoda, gubici, rast troškova, manja proizvodnja, otpuštanja, pad tražnje, problemi s likvidnošću, pogoršanje tržišnih uslova, inflatorni pritisci, regulatorni rizici ili negativne posledice po kompaniju, sektor ili ekonomiju.
- neutral: činjenična ili administrativna informacija bez jasno povoljnog ili nepovoljnog poslovno-finansijskog efekta, npr. događaj, imenovanje, izveštaj, ugovor ili tržišna aktivnost bez vrednosnog usmerenja.

Najvažnije pravilo: ako ne postoji poređenje, promena ili očigledno pozitivno/negativno značenje, oznaka je neutral. Sama brojčana vrednost nije dovoljna. Primer: „U ovoj godini neto dobit je 50 miliona.” je neutral, dok je „Neto dobit je porasla na 50 miliona.” positive.
"""

class PermanentError(RuntimeError):
    """Greška kod koje ponavljanje ne pomaže (npr. 401/403/404 - nalog nema pristup modelu)."""

class RequestRateLimiter:
    """Održava minimalni razmak između početaka zahteva jednog modela."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("Broj zahteva u minuti mora biti veći od nule.")
        self.interval = 60.0 / requests_per_minute
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled_at = max(now, self.next_request_at)
            self.next_request_at = scheduled_at + self.interval
        delay = scheduled_at - now
        if delay > 0:
            time.sleep(delay)


class ProgressLogger:
    """Ispisuje isti, odmah flush-ovan napredak u terminal i log fajl."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, message: str, *, error: bool = False) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} | {message}"
        with self._lock:
            print(line, file=sys.stderr if error else sys.stdout, flush=True)
            print(line, file=self._file, flush=True)

    def close(self) -> None:
        self._file.close()


def text_excerpt(text: str, limit: int) -> str:
    """Jednolinijski prikaz teksta za log; limit 0 znaci ceo tekst."""
    normalized = " ".join(text.split())
    if limit == 0 or len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def load_dotenv(path: Path) -> None:
    """Učitava jednostavan KEY=value .env fajl bez dodatne biblioteke."""
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Neispravan .env red {line_number}: očekuje se KEY=value.")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"Neispravan .env red {line_number}: prazan ključ.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_input(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Ulazni fajl ne postoji: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Ulaz nije ispravan JSON: {error}") from error

    if not isinstance(payload, list):
        raise ValueError("Ulazni JSON mora biti lista objekata.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Zapis {index} mora biti JSON objekat.")
        text = item.get("tekst")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Zapis {index} mora imati neprazno polje 'tekst'.")
        source_sentiment = item.get("sentiment")
        if not isinstance(source_sentiment, str) or source_sentiment.strip().lower() not in VALID_SENTIMENTS:
            raise ValueError(
                f"Zapis {index} mora imati polje 'sentiment' sa vrednošću "
                "positive, negative ili neutral."
            )
        # Zadržavamo sva polja ulaza neizmenjena, a dodajemo naš redni identifikator.
        record = dict(item)
        record["id"] = index
        records.append(record)
    return records


def extract_sentiment(model_response: str) -> str:
    """Prihvata i retke odgovore poput 'Sentiment: neutral', ali strogo validira oznaku."""
    normalized = model_response.strip().lower()
    if normalized in VALID_SENTIMENTS:
        return normalized
    words = re.findall(r"\b(?:positive|negative|neutral)\b", normalized)
    if len(words) == 1:
        return words[0]
    raise ValueError(f"Model nije vratio važeću oznaku: {model_response!r}")


def classify(
    session: requests.Session,
    api_key: str,
    model: str,
    text: str,
    limiter: RequestRateLimiter,
    retries: int,
    timeout_seconds: float,
    record_id: int,
    logger: ProgressLogger,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Klasifikuj sledeći tekst:\n\n{text}"},
        ],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    for attempt in range(retries + 1):
        limiter.wait()
        try:
            response = session.post(API_URL, headers=headers, json=payload, timeout=timeout_seconds)
            if response.status_code in (401, 403, 404):
                raise PermanentError(
                    f"{model}: trajna greška (HTTP {response.status_code}) - "
                    f"verovatno nedostaje pristup modelu na nalogu: {response.text[:300]}"
                )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise requests.HTTPError(
                    f"HTTP {response.status_code}: {response.text[:500]}", response=response
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Odgovor modela nema tekstualni sadržaj.")
            return extract_sentiment(content)
        except PermanentError:
            raise  # trajna greška - ne pokušavaj ponovo unutar ove funkcije
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            if attempt == retries:
                raise RuntimeError(f"{model}: neuspešna klasifikacija: {error}") from error
            logger.log(
                f"PONOVNI POKUŠAJ | model={model} | id={record_id} | "
                f"pokušaj={attempt + 1}/{retries} | razlog={error}",
                error=True,
            )
            retry_after = 0.0
            if isinstance(error, requests.HTTPError) and error.response is not None:
                try:
                    retry_after = float(error.response.headers.get("Retry-After", 0))
                except ValueError:
                    retry_after = 0.0
            time.sleep(max(retry_after, 2**attempt))

    raise AssertionError("Nedostižan kod")


def classify_model(
    model: str,
    records: Iterable[dict[str, Any]],
    api_key: str,
    requests_per_minute: float,
    retries: int,
    timeout_seconds: float,
    logger: ProgressLogger,
    log_text_limit: int,
    final_retry_rounds: int = 2,
) -> list[tuple[int, str | None, str | None]]:
    limiter = RequestRateLimiter(requests_per_minute)
    records = list(records)
    # (prediction, error, permanent) po id-ju zapisa
    state: dict[int, tuple[str | None, str | None, bool]] = {}

    with requests.Session() as session:

        def run_pass(target_records: list[dict[str, Any]], pass_label: str) -> None:
            for record in target_records:
                record_id = record["id"]
                try:
                    sentiment = classify(
                        session, api_key, model, record["tekst"], limiter, retries, timeout_seconds,
                        record_id, logger,
                    )
                    state[record_id] = (sentiment, None, False)
                    gold = record["sentiment"].strip().lower()
                    logger.log(
                        f"REZULTAT | model={model} | id={record_id} | sentiment={sentiment} | "
                        f"gold={gold} | slaže_se={sentiment == gold} | krug={pass_label} | "
                        f"tekst={text_excerpt(record['tekst'], log_text_limit)!r}"
                    )
                except PermanentError as error:
                    state[record_id] = (None, str(error), True)
                    logger.log(
                        f"GREŠKA (TRAJNA) | model={model} | id={record_id} | razlog={error} | "
                        f"tekst={text_excerpt(record['tekst'], log_text_limit)!r}",
                        error=True,
                    )
                except RuntimeError as error:
                    state[record_id] = (None, str(error), False)
                    logger.log(
                        f"GREŠKA | model={model} | id={record_id} | razlog={error} | krug={pass_label} | "
                        f"tekst={text_excerpt(record['tekst'], log_text_limit)!r}",
                        error=True,
                    )

        run_pass(records, "prvi prolaz")

        for round_number in range(1, final_retry_rounds + 1):
            failed_records = [
                record for record in records
                if state[record["id"]][1] is not None and not state[record["id"]][2]
            ]
            if not failed_records:
                break
            logger.log(
                f"DODATNI KRUG | model={model} | krug={round_number}/{final_retry_rounds} | "
                f"preostalo_za_ponovni_pokušaj={len(failed_records)}"
            )
            run_pass(failed_records, f"dodatni krug {round_number}")

    logger.log(f"MODEL ZAVRŠEN | model={model} | obrađeno={len(state)}")
    return [(record["id"], *state[record["id"]][:2]) for record in records]


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("Navedite barem jedan model.")
    return models


def model_filename(model: str) -> str:
    """Bezbedno ime fajla koje je čitljivo i stabilno za jedan model."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_") + ".json"


def make_model_output(
    model: str,
    records: Iterable[dict[str, Any]],
    results: Iterable[tuple[int, str | None, str | None]],
) -> list[dict[str, Any]]:
    results_by_id = {record_id: (prediction, error) for record_id, prediction, error in results}
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        prediction, error = results_by_id[record["id"]]
        item["model"] = model
        item["model_sentiment"] = prediction
        item["agrees_with_initial_sentiment"] = (
            prediction == record["sentiment"].strip().lower() if prediction else None
        )
        if error:
            item["model_error"] = error
        output.append(item)
    return output


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Finansijski sentiment preko NVIDIA NIM API-ja.")
    parser.add_argument("input", type=Path, help="Ulazni JSON: [{\"tekst\": \"...\"}]")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("sentiment_rezultati"),
        help="Direktorijum za tri pojedinačna i jedan agregirani JSON (podrazumevano: sentiment_rezultati).",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Podrazumevano: .env")
    parser.add_argument(
        "--models", type=parse_models, default=DEFAULT_MODELS,
        help="Modeli odvojeni zarezom; podrazumevano su sva tri tražena modela.",
    )
    parser.add_argument("--rpm", type=float, default=5, help="Maksimalno zahteva/minut po modelu (podrazumevano: 5).")
    parser.add_argument("--retries", type=int, default=3, help="Ponovni pokušaji za 429/5xx/mrežne greške (podrazumevano: 3).")
    parser.add_argument(
        "--final-retry-rounds", type=int, default=3,
        help="Dodatni krugovi na kraju obrade za zapise koji su i dalje neuspešni posle svih pokušaja "
             "(podrazumevano: 3; ne odnosi se na trajne greške poput 404).",
    )
    parser.add_argument("--timeout", type=float, default=90, help="Timeout po zahtevu u sekundama (podrazumevano: 90).")
    parser.add_argument(
        "--log-file", type=Path,
        help="Fajl za praćenje napretka; podrazumevano: jedinstven fajl u /tmp.",
    )
    parser.add_argument(
        "--log-text-limit", type=int, default=180,
        help="Maksimalan broj znakova teksta u svakoj log stavci; 0 = ceo tekst (podrazumevano: 180).",
    )
    args = parser.parse_args()

    if args.retries < 0 or args.timeout <= 0 or args.log_text_limit < 0:
        parser.error("--retries mora biti >= 0, --timeout > 0, a --log-text-limit >= 0.")
    try:
        load_dotenv(args.env_file)
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY nije pronađen. Dodajte ga u .env ili postavite promenljivu okruženja."
            )
        records = read_input(args.input)
        RequestRateLimiter(args.rpm)
    except ValueError as error:
        parser.error(str(error))

    default_log_name = f"nvidia_sentiment_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
    log_path = args.log_file or Path(gettempdir()) / default_log_name
    logger = ProgressLogger(log_path)
    try:
        logger.log(
            f"POČETAK | zapisa={len(records)} | modeli={', '.join(args.models)} | "
            f"rpm_po_modelu={args.rpm} | izlaz={args.output_dir} | log={log_path}"
        )
        model_results: dict[str, list[tuple[int, str | None, str | None]]] = {}

        with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
            futures = {
                executor.submit(
                    classify_model, model, records, api_key, args.rpm, args.retries, args.timeout,
                    logger, args.log_text_limit, args.final_retry_rounds,
                ): model
                for model in args.models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    model_results[model] = future.result()
                except Exception as error:  # zaštita od neočekivane greške u jednoj niti
                    logger.log(f"PREKID MODELA | model={model} | razlog={error}", error=True)
                    model_results[model] = [
                        (record["id"], None, f"{model}: prekid obrade: {error}") for record in records
                    ]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results_by_model = {
            model: {record_id: (prediction, error) for record_id, prediction, error in results}
            for model, results in model_results.items()
        }
        aggregate = []
        for record in records:
            item = dict(record)
            item["model_predictions"] = {}
            for model in args.models:
                prediction, error = results_by_model[model][record["id"]]
                item["model_predictions"][model] = {
                    "sentiment": prediction,
                    "agrees_with_initial_sentiment": (
                        prediction == record["sentiment"].strip().lower() if prediction else None
                    ),
                }
                if error:
                    item["model_predictions"][model]["error"] = error
            aggregate.append(item)

        for model in args.models:
            path = args.output_dir / model_filename(model)
            write_json(path, make_model_output(model, records, model_results[model]))
            logger.log(f"SAČUVANO | model={model} | fajl={path}")

        aggregate_path = args.output_dir / "agregirani_rezultati.json"
        write_json(aggregate_path, aggregate)
        logger.log(f"SAČUVANO | agregirani_fajl={aggregate_path}")
        logger.log("ZAVRŠENO | svi modeli i izlazni fajlovi su obrađeni")
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
