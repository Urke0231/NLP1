#!/usr/bin/env python3
"""Klasifikacija finansijskog sentimenta kroz NVIDIA NIM modele.

Svaki model obrađuje ulazne zapise redom, uz zasebno ograničenje broja
zahteva. Tri modela se izvršavaju paralelno, pa spor ili neuspešan zahtev
jednog modela ne blokira preostala dva.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable

import requests


API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODELS = (
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
    # "moonshotai/kimi-k2.6",
    "mistralai/mistral-medium-3.5-128b",
    "deepseek-ai/deepseek-v4-flash",
)
DEFAULT_AZURE_MODELS = ("gpt-5-mini",)
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

SR_ZERO_PROMPT = """Ti si strogi klasifikator finansijskog sentimenta za tekstove na srpskom jeziku.

Vrati ISKLJUČIVO jednu malu englesku reč: positive, negative ili neutral. Ne dodaj objašnjenje, interpunkciju ni JSON.
"""

SR_DEFINITIONS_PROMPT = """Ti si strogi klasifikator finansijskog sentimenta za tekstove na srpskom jeziku.

Vrati ISKLJUČIVO jednu malu englesku reč: positive, negative ili neutral. Ne dodaj objašnjenje, interpunkciju ni JSON.

Pravila:
- positive: povoljan poslovni, finansijski ili ekonomski razvoj, npr. rast prihoda, dobiti, prodaje, izvoza, investicija, zaposlenosti, tržišnog učešća, kreditnog rejtinga ili poverenja potrošača; najava poboljšanja poslovanja, stabilizacije tržišta ili uspešne strategije.
- negative: nepovoljan poslovni, finansijski ili ekonomski razvoj, npr. pad prihoda, gubici, rast troškova, manja proizvodnja, otpuštanja, pad tražnje, problemi s likvidnošću, pogoršanje tržišnih uslova, inflatorni pritisci, regulatorni rizici ili negativne posledice po kompaniju, sektor ili ekonomiju.
- neutral: činjenična ili administrativna informacija bez jasno povoljnog ili nepovoljnog poslovno-finansijskog efekta, npr. događaj, imenovanje, izveštaj, ugovor ili tržišna aktivnost bez vrednosnog usmerenja.

Najvažnije pravilo: ako ne postoji poređenje, promena ili očigledno pozitivno/negativno značenje, oznaka je neutral. Sama brojčana vrednost nije dovoljna. Primer: „U ovoj godini neto dobit je 50 miliona.” je neutral, dok je „Neto dobit je porasla na 50 miliona.” positive.
"""

FEW_SHOT_EXAMPLES = """Primeri:
Tekst: Prihodi kompanije porasli su za 15% u odnosu na prethodnu godinu.
Oznaka: positive

Tekst: Izvoz je povećan, dok je trgovinski deficit smanjen.
Oznaka: positive

Tekst: Kompanija je otvorila novi pogon i zaposlila 200 radnika.
Oznaka: positive

Tekst: Neto dobit pala je za 20% usled rasta troškova.
Oznaka: negative

Tekst: Kompanija je otpustila 300 radnika zbog pada tražnje.
Oznaka: negative

Tekst: Inflacija je ubrzana, dok je realna kupovna moć stanovništva smanjena.
Oznaka: negative

Tekst: Neto dobit u ovoj godini iznosi 50 miliona dinara.
Oznaka: neutral

Tekst: Kompanija je objavila godišnji finansijski izveštaj.
Oznaka: neutral

Tekst: Potpisan je ugovor o saradnji između dve kompanije.
Oznaka: neutral"""

SR_FEW_SHOT_PROMPT = f"""{SR_ZERO_PROMPT.rstrip()}

Koristi sledeće označene primere kao smernice. Zatim klasifikuj novi tekst prema istom kriterijumu.

{FEW_SHOT_EXAMPLES}
"""

EN_ZERO_PROMPT = """You are a strict financial sentiment classifier for Serbian-language texts.

Return EXACTLY one lowercase English word: positive, negative, or neutral. Do not add an explanation, punctuation, or JSON.
"""

EN_DEFINITIONS_PROMPT = """You are a strict financial sentiment classifier for Serbian-language texts.

Return EXACTLY one lowercase English word: positive, negative, or neutral. Do not add an explanation, punctuation, or JSON.

Rules:
- positive: a favorable business, financial, or economic development, such as growth in revenue, profit, sales, exports, investment, employment, market share, credit rating, or consumer confidence; an announced improvement in operations, market stabilization, or a successful strategy.
- negative: an unfavorable business, financial, or economic development, such as declining revenue, losses, rising costs, lower production, layoffs, falling demand, liquidity problems, worsening market conditions, inflationary pressures, regulatory risks, or negative consequences for a company, sector, or economy.
- neutral: factual or administrative information without a clearly favorable or unfavorable business-financial effect, such as an event, appointment, report, contract, or market activity without a directional evaluation.

Most important rule: if there is no comparison, change, or clearly positive/negative meaning, the label is neutral. A numerical value alone is insufficient. Example: “This year, net profit is 50 million.” is neutral, while “Net profit increased to 50 million.” is positive.
"""

EN_FEW_SHOT_PROMPT = f"""{EN_ZERO_PROMPT.rstrip()}

Use the following labeled examples as guidance. Then classify the new Serbian text using the same criterion.

{FEW_SHOT_EXAMPLES.replace('Primeri:', 'Examples:').replace('Tekst:', 'Serbian text:').replace('Oznaka:', 'Label:')}
"""

PROMPT_CONFIGS: dict[str, dict[str, str]] = {
    "sr_zero": {
        "language": "srpski", "type": "zero", "system": SR_ZERO_PROMPT,
        "user_template": "Klasifikuj sledeći tekst:\n\n{tekst}",
    },
    "sr_def": {
        "language": "srpski", "type": "definitions", "system": SR_DEFINITIONS_PROMPT,
        "user_template": "Klasifikuj sledeći tekst:\n\n{tekst}",
    },
    "sr_few": {
        "language": "srpski", "type": "few_shot", "system": SR_FEW_SHOT_PROMPT,
        "user_template": "Klasifikuj sledeći tekst:\n\n{tekst}",
    },
    "en_zero": {
        "language": "engleski", "type": "zero", "system": EN_ZERO_PROMPT,
        "user_template": "Classify the following Serbian text:\n\n{tekst}",
    },
    "en_def": {
        "language": "engleski", "type": "definitions", "system": EN_DEFINITIONS_PROMPT,
        "user_template": "Classify the following Serbian text:\n\n{tekst}",
    },
    "en_few": {
        "language": "engleski", "type": "few_shot", "system": EN_FEW_SHOT_PROMPT,
        "user_template": "Classify the following Serbian text:\n\n{tekst}",
    },
}
ALL_PROMPT_IDS = tuple(PROMPT_CONFIGS)

class PermanentError(RuntimeError):
    """Greška kod koje ponavljanje ne pomaže (npr. 401/403/404 - nalog nema pristup modelu)."""


def azure_chat_url(endpoint: str) -> str:
    """Normalizuje Azure OpenAI resource/base ili puni API endpoint."""
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT ne sme biti prazan.")
    if "/responses" in endpoint:
        return endpoint
    if "/chat/completions" in endpoint:
        return endpoint
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/openai/v1/chat/completions"

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


def extract_response_content(body: dict[str, Any], responses_api: bool) -> str:
    """Čita tekst iz Chat Completions ili Responses API odgovora."""
    if not responses_api:
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, str) and content.strip():
            return content
        finish_reason = body["choices"][0].get("finish_reason")
        usage = body.get("usage")
        raise ValueError(
            "Chat Completions odgovor nema završni tekst "
            f"(finish_reason={finish_reason!r}, usage={usage!r})."
        )

    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    texts: list[str] = []
    for output_item in body.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for part in output_item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    if not texts:
        raise ValueError(
            "Responses API odgovor nema završni tekst "
            f"(status={body.get('status')!r}, "
            f"incomplete_details={body.get('incomplete_details')!r}, "
            f"usage={body.get('usage')!r})."
        )
    return "".join(texts)


def classify(
    session: requests.Session,
    api_key: str,
    provider: str,
    api_url: str,
    model: str,
    prompt_config: dict[str, str],
    text: str,
    limiter: RequestRateLimiter,
    retries: int,
    timeout_seconds: float,
    azure_initial_token_budget: int,
    azure_max_token_budget: int,
    record_id: int,
    logger: ProgressLogger,
) -> str:
    responses_api = provider == "azure" and "/responses" in api_url
    user_message = prompt_config["user_template"].format(tekst=text)
    if responses_api:
        payload = {
            "model": model,
            "instructions": prompt_config["system"],
            "input": user_message,
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
    if provider == "azure":
        # GPT-5 familija ne prihvata uvek temperature=0.
        headers = {"api-key": api_key, "Accept": "application/json"}
    else:
        payload["temperature"] = 0
        payload["max_tokens"] = 8
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    for attempt in range(retries + 1):
        if provider == "azure":
            # Budžet obuhvata interne reasoning tokene i raste samo kada je potreban novi pokušaj.
            token_budget = min(azure_initial_token_budget * (2**attempt), azure_max_token_budget)
            # Poslednji pokušaj rezerviše prioritet završnoj oznaci umesto dodatnom rezonovanju.
            reasoning_effort = "minimal" if attempt == retries else "medium"
            if responses_api:
                payload["max_output_tokens"] = token_budget
                payload["reasoning"] = {"effort": reasoning_effort}
            else:
                payload["max_completion_tokens"] = token_budget
                payload["reasoning_effort"] = reasoning_effort
        limiter.wait()
        try:
            response = session.post(api_url, headers=headers, json=payload, timeout=timeout_seconds)
            if response.status_code in (400, 401, 403, 404):
                raise PermanentError(
                    f"{model}: trajna greška (HTTP {response.status_code}) - "
                    f"proverite endpoint, deployment, pristup i parametre: {response.text[:500]}"
                )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise requests.HTTPError(
                    f"HTTP {response.status_code}: {response.text[:500]}", response=response
                )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("API odgovor nije JSON objekat.")
            content = extract_response_content(body, responses_api)
            return extract_sentiment(content)
        except PermanentError:
            raise  # trajna greška - ne pokušavaj ponovo unutar ove funkcije
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            if attempt == retries:
                raise RuntimeError(f"{model}: neuspešna klasifikacija: {error}") from error
            next_budget: int | str = "n/a"
            if provider == "azure":
                next_budget = min(
                    azure_initial_token_budget * (2 ** (attempt + 1)), azure_max_token_budget
                )
            logger.log(
                f"PONOVNI POKUŠAJ | model={model} | id={record_id} | "
                f"pokušaj={attempt + 1}/{retries} | "
                f"sledeći_token_budžet={next_budget} | razlog={error}",
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
    provider: str,
    api_url: str,
    model: str,
    prompt_config: dict[str, str],
    records: Iterable[dict[str, Any]],
    api_key: str,
    requests_per_minute: float,
    workers_per_model: int,
    retries: int,
    timeout_seconds: float,
    azure_initial_token_budget: int,
    azure_max_token_budget: int,
    logger: ProgressLogger,
    log_text_limit: int,
    initial_results: dict[int, str],
    result_callback: Callable[[str, int, str | None, str | None, bool], None],
    final_retry_rounds: int = 2,
) -> list[tuple[int, str | None, str | None]]:
    limiter = RequestRateLimiter(requests_per_minute)
    records = list(records)
    # (prediction, error, permanent) po id-ju zapisa
    state: dict[int, tuple[str | None, str | None, bool]] = {
        record_id: (sentiment, None, False)
        for record_id, sentiment in initial_results.items()
    }
    pending_records = [record for record in records if record["id"] not in state]
    if initial_results:
        logger.log(f"NASTAVAK | model={model} | preskočeno_uspešnih={len(initial_results)}")

    thread_local = threading.local()
    sessions: list[requests.Session] = []
    sessions_lock = threading.Lock()

    def get_session() -> requests.Session:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = requests.Session()
            thread_local.session = session
            with sessions_lock:
                sessions.append(session)
        return session

    def process_record(record: dict[str, Any]) -> str:
        return classify(
            get_session(), api_key, provider, api_url, model, prompt_config,
            record["tekst"], limiter, retries, timeout_seconds,
            azure_initial_token_budget, azure_max_token_budget, record["id"], logger,
        )

    def run_pass(target_records: list[dict[str, Any]], pass_label: str) -> None:
        with ThreadPoolExecutor(max_workers=workers_per_model) as executor:
            futures = {executor.submit(process_record, record): record for record in target_records}
            for future in as_completed(futures):
                record = futures[future]
                record_id = record["id"]
                try:
                    sentiment = future.result()
                    state[record_id] = (sentiment, None, False)
                    result_callback(model, record_id, sentiment, None, False)
                    gold = record["sentiment"].strip().lower()
                    logger.log(
                        f"REZULTAT | model={model} | id={record_id} | sentiment={sentiment} | "
                        f"gold={gold} | slaže_se={sentiment == gold} | krug={pass_label} | "
                        f"tekst={text_excerpt(record['tekst'], log_text_limit)!r}"
                    )
                except PermanentError as error:
                    state[record_id] = (None, str(error), True)
                    result_callback(model, record_id, None, str(error), True)
                    logger.log(
                        f"GREŠKA (TRAJNA) | model={model} | id={record_id} | razlog={error} | "
                        f"tekst={text_excerpt(record['tekst'], log_text_limit)!r}",
                        error=True,
                    )
                except RuntimeError as error:
                    state[record_id] = (None, str(error), False)
                    result_callback(model, record_id, None, str(error), False)
                    logger.log(
                        f"GREŠKA | model={model} | id={record_id} | razlog={error} | "
                        f"krug={pass_label} | tekst={text_excerpt(record['tekst'], log_text_limit)!r}",
                        error=True,
                    )

    try:
        run_pass(pending_records, "prvi prolaz")

        for round_number in range(1, final_retry_rounds + 1):
            failed_records = [
                record for record in pending_records
                if state[record["id"]][1] is not None and not state[record["id"]][2]
            ]
            if not failed_records:
                break
            logger.log(
                f"DODATNI KRUG | model={model} | krug={round_number}/{final_retry_rounds} | "
                f"preostalo_za_ponovni_pokušaj={len(failed_records)}"
            )
            run_pass(failed_records, f"dodatni krug {round_number}")
    finally:
        for session in sessions:
            session.close()

    logger.log(f"MODEL ZAVRŠEN | model={model} | obrađeno={len(state)}")
    return [(record["id"], *state[record["id"]][:2]) for record in records]


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("Navedite barem jedan model.")
    return models


def parse_prompts(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if requested == ("all",):
        return ALL_PROMPT_IDS
    if not requested:
        raise argparse.ArgumentTypeError("Navedite prompt ID ili all.")
    unknown = [prompt_id for prompt_id in requested if prompt_id not in PROMPT_CONFIGS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Nepoznati promptovi: {', '.join(unknown)}. Dozvoljeno: "
            f"{', '.join(ALL_PROMPT_IDS)} ili all."
        )
    return tuple(dict.fromkeys(requested))


def prompt_hash(prompt_id: str) -> str:
    config = PROMPT_CONFIGS[prompt_id]
    canonical = json.dumps(
        {"system": config["system"], "user_template": config["user_template"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_filename(model: str) -> str:
    """Bezbedno ime fajla koje je čitljivo i stabilno za jedan model."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_") + ".json"


def make_model_output(
    provider: str,
    model: str,
    prompt_id: str,
    current_prompt_hash: str,
    records: Iterable[dict[str, Any]],
    results: Iterable[tuple[int, str | None, str | None]],
) -> list[dict[str, Any]]:
    results_by_id = {record_id: (prediction, error) for record_id, prediction, error in results}
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        prediction, error = results_by_id[record["id"]]
        item["provider"] = provider
        item["model"] = model
        item["prompt_id"] = prompt_id
        item["prompt_hash"] = current_prompt_hash
        item["model_sentiment"] = prediction
        item["agrees_with_initial_sentiment"] = (
            prediction == record["sentiment"].strip().lower() if prediction else None
        )
        if error:
            item["model_error"] = error
        output.append(item)
    return output


def write_json(path: Path, data: Any, replace_retries: int = 8) -> None:
    """Atomski menja JSON i prevazilazi kratkotrajno Windows zaključavanje fajla."""
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(replace_retries + 1):
            try:
                os.replace(temporary_path, path)
                return
            except PermissionError:
                if attempt == replace_retries:
                    raise
                time.sleep(min(0.05 * (2**attempt), 0.5))
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


class IncrementalOutputStore:
    """Dnevnik i JSON snimci koji se osvežavaju odmah posle svakog odgovora."""

    def __init__(
        self,
        output_dir: Path,
        provider: str,
        models: tuple[str, ...],
        prompt_id: str,
        records: list[dict[str, Any]],
        resume: bool,
        logger: ProgressLogger,
    ) -> None:
        self.output_dir = output_dir
        self.provider = provider
        self.models = models
        self.prompt_id = prompt_id
        self.prompt_config = PROMPT_CONFIGS[prompt_id]
        self.prompt_hash = prompt_hash(prompt_id)
        self.records = records
        self.logger = logger
        self.records_by_id = {record["id"]: record for record in records}
        self.record_hashes = {
            record["id"]: hashlib.sha256(
                json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for record in records
        }
        self.state: dict[str, dict[int, tuple[str | None, str | None]]] = {
            model: {record["id"]: (None, None) for record in records}
            for model in models
        }
        self._lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.output_dir / "prompt_config.json",
            {
                "prompt_id": self.prompt_id,
                "prompt_hash": self.prompt_hash,
                "language": self.prompt_config["language"],
                "type": self.prompt_config["type"],
                "system": self.prompt_config["system"],
                "user_template": self.prompt_config["user_template"],
            },
        )

        if resume:
            self._load_existing_outputs()
            self._replay_journals()
        else:
            for model in models:
                self.journal_path(model).write_text("", encoding="utf-8")

        # Svi standardni izlazi postoje pre prvog API zahteva.
        for model in models:
            self.journal_path(model).touch(exist_ok=True)
            try:
                self._write_model_snapshot(model)
            except OSError as snapshot_error:
                self.logger.log(
                    f"UPOZORENJE | početni modelski snapshot je zaključan; "
                    f"nastavljam iz journal-a | model={model} | razlog={snapshot_error}",
                    error=True,
                )
        try:
            self._write_aggregate_snapshot()
        except OSError as snapshot_error:
            self.logger.log(
                f"UPOZORENJE | početni agregirani snapshot je zaključan; "
                f"nastavljam iz journal-a | razlog={snapshot_error}",
                error=True,
            )

    def model_path(self, model: str) -> Path:
        return self.output_dir / model_filename(model)

    def journal_path(self, model: str) -> Path:
        return self.output_dir / (model_filename(model).removesuffix(".json") + ".journal.jsonl")

    def _valid_record(self, item: dict[str, Any], model: str) -> bool:
        record_id = item.get("id")
        record = self.records_by_id.get(record_id)
        return bool(
            record
            and item.get("tekst") == record.get("tekst")
            and item.get("sentiment") == record.get("sentiment")
            and item.get("model") == model
            and item.get("provider", self.provider) == self.provider
            and item.get("prompt_id") == self.prompt_id
            and item.get("prompt_hash") == self.prompt_hash
        )

    def _load_existing_outputs(self) -> None:
        for model in self.models:
            path = self.model_path(model)
            if not path.exists():
                continue
            try:
                items = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or not self._valid_record(item, model):
                        continue
                    prediction = item.get("model_sentiment")
                    if prediction in VALID_SENTIMENTS:
                        self.state[model][item["id"]] = (prediction, None)
            except (OSError, json.JSONDecodeError):
                continue

    def _replay_journals(self) -> None:
        for model in self.models:
            path = self.journal_path(model)
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                    record_id = event["id"]
                    if (
                        event.get("provider") != self.provider
                        or event.get("model") != model
                        or event.get("prompt_id") != self.prompt_id
                        or event.get("prompt_hash") != self.prompt_hash
                        or event.get("record_hash") != self.record_hashes.get(record_id)
                    ):
                        continue
                    prediction = event.get("prediction")
                    error = event.get("error")
                    if prediction in VALID_SENTIMENTS or isinstance(error, str):
                        self.state[model][record_id] = (prediction, error)
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue

    def successful_results(self, model: str) -> dict[int, str]:
        return {
            record_id: prediction
            for record_id, (prediction, _) in self.state[model].items()
            if prediction in VALID_SENTIMENTS
        }

    def _model_results(self, model: str) -> list[tuple[int, str | None, str | None]]:
        return [
            (record["id"], *self.state[model][record["id"]])
            for record in self.records
        ]

    def _write_model_snapshot(self, model: str) -> None:
        write_json(
            self.model_path(model),
            make_model_output(
                self.provider, model, self.prompt_id, self.prompt_hash,
                self.records, self._model_results(model),
            ),
        )

    def _write_aggregate_snapshot(self) -> None:
        aggregate: list[dict[str, Any]] = []
        for record in self.records:
            item = dict(record)
            item["provider"] = self.provider
            item["prompt_id"] = self.prompt_id
            item["prompt_hash"] = self.prompt_hash
            item["model_predictions"] = {}
            for model in self.models:
                prediction, error = self.state[model][record["id"]]
                model_result: dict[str, Any] = {
                    "sentiment": prediction,
                    "agrees_with_initial_sentiment": (
                        prediction == record["sentiment"].strip().lower() if prediction else None
                    ),
                }
                if error:
                    model_result["error"] = error
                item["model_predictions"][model] = model_result
            aggregate.append(item)
        write_json(self.output_dir / "agregirani_rezultati.json", aggregate)

    def update(
        self,
        model: str,
        record_id: int,
        prediction: str | None,
        error: str | None,
        permanent: bool,
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": self.provider,
            "model": model,
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "id": record_id,
            "record_hash": self.record_hashes[record_id],
            "prediction": prediction,
            "error": error,
            "permanent": permanent,
        }
        with self._lock:
            # Dnevnik ide prvi: čak i pad između dva upisa može da se oporavi pri restartu.
            with self.journal_path(model).open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(event, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            self.state[model][record_id] = (prediction, error)
            try:
                self._write_model_snapshot(model)
            except OSError as snapshot_error:
                self.logger.log(
                    f"UPOZORENJE | modelski JSON je privremeno zaključan; "
                    f"journal je sačuvan i snapshot će biti ponovljen | model={model} | "
                    f"id={record_id} | razlog={snapshot_error}",
                    error=True,
                )
            try:
                self._write_aggregate_snapshot()
            except OSError as snapshot_error:
                self.logger.log(
                    f"UPOZORENJE | agregirani JSON je privremeno zaključan; "
                    f"journal je sačuvan i snapshot će biti ponovljen | model={model} | "
                    f"id={record_id} | razlog={snapshot_error}",
                    error=True,
                )


def run_prompt_classification(
    args: argparse.Namespace,
    prompt_id: str,
    output_dir: Path,
    records: list[dict[str, Any]],
    api_key: str,
    api_url: str,
    logger: ProgressLogger,
) -> IncrementalOutputStore:
    """Obrađuje jedan prompt; modeli su paralelni, dok se promptovi pozivaju redom."""
    prompt_config = PROMPT_CONFIGS[prompt_id]
    logger.log(
        f"PROMPT POČETAK | prompt={prompt_id} | jezik={prompt_config['language']} | "
        f"tip={prompt_config['type']} | hash={prompt_hash(prompt_id)} | izlaz={output_dir}"
    )
    store = IncrementalOutputStore(
        output_dir, args.provider, args.models, prompt_id, records,
        resume=not args.no_resume, logger=logger,
    )
    logger.log(
        f"IZLAZI INICIJALIZOVANI | prompt={prompt_id} | svaki rezultat se odmah trajno upisuje"
    )

    executor = ThreadPoolExecutor(max_workers=len(args.models))
    model_failures: list[str] = []
    try:
        futures = {
            executor.submit(
                classify_model, args.provider, api_url, model, prompt_config,
                records, api_key, args.rpm, args.workers_per_model, args.retries,
                args.timeout, args.azure_initial_token_budget, args.azure_max_token_budget,
                logger, args.log_text_limit, store.successful_results(model), store.update,
                args.final_retry_rounds,
            ): model
            for model in args.models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as error:  # zaštita od neočekivane greške u jednoj niti
                model_failures.append(f"{model}: {error}")
                logger.log(
                    f"PREKID MODELA | prompt={prompt_id} | model={model} | razlog={error}",
                    error=True,
                )
    except KeyboardInterrupt:
        logger.log(
            "PREKINUTO | Ctrl+C | sačuvani rezultati biće automatski nastavljeni pri sledećem pokretanju",
            error=True,
        )
        executor.shutdown(wait=False, cancel_futures=True)
        logger.close()
        os._exit(130)
    else:
        executor.shutdown(wait=True)

    if model_failures:
        raise RuntimeError(
            f"Prompt {prompt_id} nije kompletiran: {'; '.join(model_failures)}"
        )

    for model in args.models:
        logger.log(
            f"SAČUVANO | prompt={prompt_id} | model={model} | fajl={store.model_path(model)}"
        )
    logger.log(
        f"PROMPT ZAVRŠEN | prompt={prompt_id} | "
        f"agregirani_fajl={output_dir / 'agregirani_rezultati.json'}"
    )
    return store


def run_visualization_analysis(
    output_root: Path,
    prompt_outputs: list[tuple[str, Path]],
    models: tuple[str, ...],
    provider: str,
    logger: ProgressLogger,
) -> None:
    """Pokreće pojedinačnu, zajedničku i six-prompt analizu svih agregata."""
    script_dir = Path(__file__).resolve().parent
    analysis_script = script_dir / "claudeNLP1" / "04_dekoderski_modeli.py"
    if not analysis_script.exists():
        raise RuntimeError(f"Skripta za analizu ne postoji: {analysis_script}")

    command = [
        sys.executable,
        str(analysis_script),
        "--models",
        ",".join(models),
        "--output-name",
        f"{provider}_six_prompt_analysis",
        "--output-dir",
        str((output_root / "analiza").resolve()),
    ]
    for prompt_id, aggregate_path in prompt_outputs:
        command.extend(["--result", f"{prompt_id}={aggregate_path.resolve()}"])

    logger.log(
        f"ANALIZA POČETAK | promptova={len(prompt_outputs)} | "
        f"skripta={analysis_script}"
    )
    completed = subprocess.run(command, cwd=analysis_script.parent, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Analiza je završena kodom {completed.returncode}. Agregirani rezultati su sačuvani."
        )
    logger.log(f"ANALIZA ZAVRŠENA | izlaz={output_root / 'analiza'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Finansijski sentiment preko NVIDIA ili Azure OpenAI API-ja.")
    parser.add_argument("input", type=Path, help="Ulazni JSON: [{\"tekst\": \"...\"}]")
    parser.add_argument(
        "--provider", choices=("nvidia", "azure"), default="nvidia",
        help="API provajder (podrazumevano: nvidia).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("sentiment_rezultati"),
        help="Direktorijum za pojedinačne, agregirani i journal izlaze.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Podrazumevano: .env")
    parser.add_argument(
        "--models", type=parse_models,
        help="NVIDIA modeli ili Azure deployment imena, odvojeni zarezom.",
    )
    parser.add_argument(
        "--prompts", type=parse_prompts, default=("sr_def",),
        help=(
            "Prompt ID-jevi odvojeni zarezom ili all. Dozvoljeno: "
            f"{', '.join(ALL_PROMPT_IDS)} (podrazumevano: sr_def)."
        ),
    )
    parser.add_argument(
        "--azure-endpoint",
        help="Azure OpenAI resource/base ili puni chat endpoint; inače AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--workers-per-model", type=int,
        help="Istovremeni zahtevi po modelu (podrazumevano: NVIDIA 1, Azure 8).",
    )
    parser.add_argument(
        "--rpm", type=float,
        help="Maksimalno početaka zahteva/minut po modelu (podrazumevano: NVIDIA 5, Azure 120).",
    )
    parser.add_argument(
        "--azure-initial-token-budget", type=int, default=1024,
        help="Početni Azure completion/reasoning budžet (podrazumevano: 1024).",
    )
    parser.add_argument(
        "--azure-max-token-budget", type=int, default=8192,
        help="Najveći Azure completion/reasoning budžet pri ponavljanju (podrazumevano: 8192).",
    )
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
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignoriši ranije uspešne rezultate i kreni ispočetka (journal se prazni).",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Ne pokreći automatsku evaluaciju i vizualizaciju po završetku promptova.",
    )
    args = parser.parse_args()

    if (
        args.retries < 0
        or args.final_retry_rounds < 0
        or args.timeout <= 0
        or args.log_text_limit < 0
    ):
        parser.error(
            "--retries i --final-retry-rounds moraju biti >= 0, --timeout > 0, "
            "a --log-text-limit >= 0."
        )
    try:
        load_dotenv(args.env_file)
        if args.provider == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY nije pronađen u okruženju ili .env fajlu.")
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT nije pronađen; prosledite i --azure-endpoint ili .env vrednost.")
            api_url = azure_chat_url(endpoint)
            if args.models is None:
                deployments = os.environ.get("AZURE_OPENAI_DEPLOYMENTS")
                args.models = parse_models(deployments) if deployments else DEFAULT_AZURE_MODELS
            args.workers_per_model = args.workers_per_model or 8
            args.rpm = args.rpm or 120.0
        else:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY nije pronađen u okruženju ili .env fajlu.")
            api_url = API_URL
            args.models = args.models or DEFAULT_MODELS
            args.workers_per_model = args.workers_per_model or 1
            args.rpm = args.rpm or 5.0

        if args.workers_per_model <= 0:
            raise ValueError("--workers-per-model mora biti veći od nule.")
        if (
            args.azure_initial_token_budget <= 0
            or args.azure_max_token_budget < args.azure_initial_token_budget
        ):
            raise ValueError(
                "--azure-initial-token-budget mora biti > 0, a --azure-max-token-budget "
                "mora biti jednak ili veći od početnog."
            )
        records = read_input(args.input)
        RequestRateLimiter(args.rpm)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))

    default_log_name = f"nvidia_sentiment_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
    log_path = args.log_file or Path(gettempdir()) / default_log_name
    logger = ProgressLogger(log_path)
    try:
        logger.log(
            f"POČETAK | provider={args.provider} | zapisa={len(records)} | "
            f"modeli={', '.join(args.models)} | workers_po_modelu={args.workers_per_model} | "
            f"promptovi={', '.join(args.prompts)} | "
            f"rpm_po_modelu={args.rpm} | azure_token_budžet="
            f"{args.azure_initial_token_budget}-{args.azure_max_token_budget} | "
            f"izlaz={args.output_dir} | log={log_path}"
        )
        multiple_prompts = len(args.prompts) > 1
        prompt_outputs: list[tuple[str, Path]] = []
        for prompt_id in args.prompts:
            prompt_output_dir = (
                args.output_dir / prompt_id if multiple_prompts else args.output_dir
            )
            run_prompt_classification(
                args, prompt_id, prompt_output_dir, records, api_key, api_url, logger,
            )
            prompt_outputs.append(
                (prompt_id, prompt_output_dir / "agregirani_rezultati.json")
            )

        if multiple_prompts and not args.skip_analysis:
            run_visualization_analysis(
                args.output_dir, prompt_outputs, args.models, args.provider, logger,
            )
        logger.log("ZAVRŠENO | svi promptovi, modeli i izlazni fajlovi su obrađeni")
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
