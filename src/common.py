"""File formats shared by the collection, preparation and analysis scripts."""

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def load_seeds(path):
    seeds = read_csv(path)
    seen = set()
    for seed in seeds:
        name = seed.get('full_name', '')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', name):
            raise ValueError(f'Invalid repository name: {name!r}')
        if name.lower() in seen or not seed.get('family') or not seed.get('reason'):
            raise ValueError(f'Duplicate seed or missing selection reason: {name}')
        seen.add(name.lower())
    if not seeds:
        raise ValueError('The seed list is empty')
    return seeds


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_int(value, field):
    if isinstance(value, bool):
        raise ValueError(f'{field} must be a positive integer')
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field} must be a positive integer: {value!r}') from None
    if str(number) != str(value) or number <= 0:
        raise ValueError(f'{field} must be a positive integer: {value!r}')
    return number

