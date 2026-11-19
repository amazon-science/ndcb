"""Locating, downloading and loading the NDCB data files.

Two files make up the benchmark:

`ndcb_scenarios.jsonl` (697 rows)
    The scenarios themselves: context, the two candidates, the decision task,
    the answer menu, and annotation metadata.

`prompts.jsonl` (1,394 rows = 697 x 2 framings)
    The **pre-rendered** evaluation prompts, byte-for-byte as sent to the six
    models in the paper.

Why ship rendered prompts rather than a prompt builder? The research prompt
builder drifted after the data was collected and can no longer regenerate what
was actually sent -- in particular it only emits the single-pick normative
variant, while 442 of the 697 scenarios used a per-candidate variant. These
files were reconstructed from the `question` field recorded in every stored
judge record, and verified identical across all six models and all three runs
(zero conflicts over 1,394 keys). Replaying them is the only way to reproduce
the paper's numbers exactly, so that is what the runner does.

Resolution order for the data directory:

1. ``$NDCB_DATA_DIR``
2. ``<repo>/data`` -- i.e. you are working from a git clone
3. ``$XDG_CACHE_HOME/ndcb`` (or ``~/.cache/ndcb``) -- where ``ndcb download``
   puts them
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional

SCENARIOS_FILE = "ndcb_scenarios.jsonl"
PROMPTS_FILE = "prompts.jsonl"
DATA_FILES = (SCENARIOS_FILE, PROMPTS_FILE)

#: Where `ndcb download` fetches the data when no URL is given: the released
#: repository's `data/` directory, pinned to the main branch.
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/amazon-science/ndcb/main/data"

N_SCENARIOS = 697
N_PROMPTS = 1394

#: SHA-256 of the released files. `ndcb download --verify` checks these, so a
#: truncated or tampered download fails loudly instead of silently changing
#: someone's reported score.
CHECKSUMS = {
    SCENARIOS_FILE: "6a2946fe4a36264f6a5df75cc082dee7ba46a1f1cdbc5989cce3cc51e115bedc",
    PROMPTS_FILE: "37bd5764b7d7e9006a954f522bc7bcfb340e4ca1ef41d83dde479f839aa2e3db",
}


# --- paths ---------------------------------------------------------------- #
def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "ndcb"


def _repo_data_dir() -> Path:
    # src/ndcb/dataset.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2] / "data"


def candidate_dirs() -> List[Path]:
    dirs = []
    env = os.environ.get("NDCB_DATA_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(_repo_data_dir())
    dirs.append(cache_dir())
    return dirs


def data_dir() -> Path:
    """First directory that holds both data files. Raises if none does."""
    for d in candidate_dirs():
        if all((d / f).is_file() for f in DATA_FILES):
            return d
    searched = "\n  ".join(str(d) for d in candidate_dirs())
    raise FileNotFoundError(
        "NDCB data files not found. Searched:\n  " + searched +
        "\n\nRun `ndcb download` to fetch them, `ndcb download --from-dir "
        "<path>` to copy from a local checkout, or set $NDCB_DATA_DIR."
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(directory: Path) -> Dict[str, bool]:
    """Map each data file to whether its SHA-256 matches the release."""
    return {f: (directory / f).is_file() and sha256(directory / f) == CHECKSUMS[f]
            for f in DATA_FILES}


# --- acquiring ------------------------------------------------------------ #
def download(dest: Optional[Path] = None, from_dir: Optional[Path] = None,
             base_url: Optional[str] = None, check: bool = True) -> Path:
    """Put the data files in `dest` (default: the user cache dir).

    `from_dir` copies from a local path -- the reliable route, and the one to
    use when you already have the repository checked out. Otherwise the files
    are fetched over HTTP from `base_url`, ``$NDCB_DATA_URL``, or the released
    repository (`DEFAULT_DATA_URL`).
    """
    dest = Path(dest) if dest else cache_dir()
    dest.mkdir(parents=True, exist_ok=True)

    if from_dir is not None:
        src = Path(from_dir).expanduser()
        for f in DATA_FILES:
            if not (src / f).is_file():
                raise FileNotFoundError(f"{src / f} does not exist")
            shutil.copy2(src / f, dest / f)
    else:
        url_base = (base_url or os.environ.get("NDCB_DATA_URL")
                    or DEFAULT_DATA_URL).rstrip("/")
        import httpx
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            for f in DATA_FILES:
                url = f"{url_base}/{f}"
                with open(dest / f, "wb") as out:
                    with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        for chunk in resp.iter_bytes():
                            out.write(chunk)

    if check:
        bad = [f for f, ok in verify(dest).items() if not ok]
        if bad:
            raise RuntimeError(
                "Checksum mismatch after download for: " + ", ".join(bad) +
                ". The files do not match the released benchmark; scores "
                "computed from them would not be comparable."
            )
    return dest


# --- loading -------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_scenarios(directory: Optional[Path] = None) -> List[Dict]:
    """The 697 scenarios, in file order."""
    d = Path(directory) if directory else data_dir()
    rows = list(_read_jsonl(d / SCENARIOS_FILE))
    if len(rows) != N_SCENARIOS:
        raise ValueError(f"expected {N_SCENARIOS} scenarios, read {len(rows)}")
    return rows


def load_prompts(directory: Optional[Path] = None) -> Dict[tuple, Dict]:
    """The rendered prompts, keyed by ``(scenario_id, framing)``."""
    d = Path(directory) if directory else data_dir()
    rows = list(_read_jsonl(d / PROMPTS_FILE))
    if len(rows) != N_PROMPTS:
        raise ValueError(f"expected {N_PROMPTS} prompts, read {len(rows)}")
    return {(r["scenario_id"], r["framing"]): r for r in rows}
