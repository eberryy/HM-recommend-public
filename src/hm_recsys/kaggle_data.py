from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv

COMPETITION = "h-and-m-personalized-fashion-recommendations"
REQUIRED_MVP_FILE = "transactions_train.csv"
TABULAR_FILES = (
    "transactions_train.csv",
    "articles.csv",
    "customers.csv",
    "sample_submission.csv",
)


@dataclass(frozen=True)
class DataStatus:
    auth_source: str | None
    raw_dir: Path
    transactions_path: Path

    @property
    def authenticated(self) -> bool:
        return self.auth_source is not None

    @property
    def ready(self) -> bool:
        return self.transactions_path.is_file()

    @property
    def tabular_ready(self) -> bool:
        return all((self.raw_dir / filename).is_file() for filename in TABULAR_FILES)

    @property
    def tabular_files(self) -> dict[str, bool]:
        return {
            filename: (self.raw_dir / filename).is_file()
            for filename in TABULAR_FILES
        }


def load_project_env(env_file: str | Path = ".env") -> Path:
    path = Path(env_file).expanduser().resolve()
    if path.is_file():
        load_dotenv(path, override=False)
    return path


def detect_auth_source() -> str | None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY"

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    if (config_dir / "access_token").is_file():
        return "access_token file"
    if (config_dir / "kaggle.json").is_file():
        return "kaggle.json"
    return None


def get_data_status(
    raw_dir: str | Path = "data/raw", env_file: str | Path = ".env"
) -> DataStatus:
    load_project_env(env_file)
    raw_path = Path(raw_dir).expanduser().resolve()
    return DataStatus(
        auth_source=detect_auth_source(),
        raw_dir=raw_path,
        transactions_path=raw_path / REQUIRED_MVP_FILE,
    )


def extract_competition_archive(archive: Path, destination: Path) -> None:
    """Extract a trusted competition archive while rejecting path traversal."""
    destination = destination.resolve()
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def _run_download_command(
    command: list[str],
    *,
    env: dict[str, str],
    attempts: int,
) -> None:
    if attempts < 1:
        raise ValueError("download_attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True, env=env)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"Kaggle download attempt {attempt}/{attempts} failed; "
                f"retrying in {delay}s. Existing partial files are preserved.",
                flush=True,
            )
            time.sleep(delay)

def extract_image_archive(
    archive: Path,
    destination: Path,
) -> dict[str, int]:
    """Extract only trusted H&M image members with resumable atomic writes."""
    archive = archive.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"competition archive not found: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    extracted_bytes = 0
    image_members = 0
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            normalised = member.filename.replace(chr(92), "/")
            if member.is_dir() or not normalised.startswith("images/"):
                continue
            image_members += 1
            target = (destination / normalised).resolve()
            if destination not in target.parents:
                raise ValueError(f"unsafe image archive member: {member.filename}")
            if target.is_file():
                if target.stat().st_size != member.file_size:
                    raise ValueError(
                        f"existing image size mismatch; refusing overwrite: {target}"
                    )
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with zipped.open(member) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise ValueError(f"extracted image size mismatch: {member.filename}")
            partial.replace(target)
            extracted += 1
            extracted_bytes += member.file_size
    if image_members == 0:
        raise ValueError(f"archive contains no images/: {archive}")
    return {
        "image_members": image_members,
        "extracted_files": extracted,
        "skipped_files": skipped,
        "extracted_bytes": extracted_bytes,
    }

def download_competition_data(
    *,
    raw_dir: str | Path = "data/raw",
    env_file: str | Path = ".env",
    force: bool = False,
    extract: bool = True,
    all_files: bool = False,
    tabular: bool = False,
    download_attempts: int = 5,
) -> DataStatus:
    status = get_data_status(raw_dir, env_file)
    if all_files and tabular:
        raise ValueError("--all-files and --tabular are mutually exclusive")
    requested_files = TABULAR_FILES if tabular else (REQUIRED_MVP_FILE,)
    requested_ready = all(
        (status.raw_dir / filename).is_file() for filename in requested_files
    )
    if requested_ready and not force and not all_files:
        return status
    if not status.authenticated:
        raise RuntimeError(
            "Kaggle credentials not found. Copy .env.example to .env and set "
            "KAGGLE_API_TOKEN, or configure ~/.kaggle/access_token or kaggle.json."
        )

    kaggle_executable = shutil.which("kaggle")
    if kaggle_executable is None:
        raise RuntimeError(
            "Kaggle CLI is not installed in this environment. Run "
            "python -m pip install -r dependency.txt."
        )

    status.raw_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    if status.auth_source in {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME/KAGGLE_KEY"}:
        config_dir = status.raw_dir.parent / ".kaggle"
        config_dir.mkdir(parents=True, exist_ok=True)
        child_env.setdefault("KAGGLE_CONFIG_DIR", str(config_dir))

    if all_files:
        command = [
            kaggle_executable,
            "competitions",
            "download",
            COMPETITION,
            "-p",
            str(status.raw_dir),
        ]
        if force:
            command.append("-o")
        _run_download_command(
            command,
            env=child_env,
            attempts=download_attempts,
        )
        archive = status.raw_dir / f"{COMPETITION}.zip"
        if extract:
            if not archive.is_file():
                raise FileNotFoundError(f"competition archive not found: {archive}")
            extract_competition_archive(archive, status.raw_dir)
    else:
        for filename in requested_files:
            destination = status.raw_dir / filename
            if destination.is_file() and not force:
                continue
            command = [
                kaggle_executable,
                "competitions",
                "download",
                COMPETITION,
                "-f",
                filename,
                "-p",
                str(status.raw_dir),
            ]
            if force:
                command.append("-o")
            _run_download_command(
                command,
                env=child_env,
                attempts=download_attempts,
            )
            archive = status.raw_dir / f"{filename}.zip"
            if extract and not destination.is_file():
                if not archive.is_file():
                    raise FileNotFoundError(
                        f"Kaggle download finished but {filename} was not found"
                    )
                extract_competition_archive(archive, status.raw_dir)

    refreshed = get_data_status(status.raw_dir, env_file)
    if extract and not all(
        (refreshed.raw_dir / filename).is_file() for filename in requested_files
    ):
        raise FileNotFoundError(
            f"requested files were not found after extraction in {status.raw_dir}"
        )
    return refreshed
