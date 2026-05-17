import json
import logging
import os
from pathlib import Path
import shutil
import zipfile

import kagglehub
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DATASET_OWNER = "sehaj1104"
DATASET_SLUG = "student-mental-health-and-burnout-dataset"
DATASET_FILE = "student_mental_health_burnout.csv"

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


class KaggleDataLoader:
    def __init__(
        self, owner=DATASET_OWNER, dataset=DATASET_SLUG, output_dir="data/raw"
    ):
        self.owner = owner
        self.dataset = dataset
        self.dataset_path = f"{owner}/{dataset}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._downloaded_dir = None

        # Подготовим REST-сессию, если есть новый токен
        load_dotenv()
        self._username = os.getenv("KAGGLE_USERNAME")
        self._key = os.getenv("KAGGLE_KEY")
        if not (self._username and self._key):
            self._load_credentials_from_json()

        self._rest_available = False
        if self._key and self._key.startswith("KGAT"):
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self._key}",
                    "User-Agent": "KaggleDataLoader/1.0",
                }
            )
            self._rest_available = True
        else:
            self._session = None

    # ── Основные методы ─────────────────────────────────────────────────────
    def download(self) -> Path:
        """
        Сначала пробуем kagglehub.
        Если не получилось — fallback через REST API.
        """
        try:
            return self._download_via_kagglehub()
        except Exception as e:
            logger.warning(
                "kagglehub download failed: %s",
                e,
            )
            print("Switching to REST API fallback...")
        return self._download_via_rest()

    def _download_via_kagglehub(self) -> Path:
        """
        Основной способ загрузки через kagglehub.
        """
        target = self.output_dir / DATASET_FILE

        if target.exists():
            print(f"✓ Dataset already exists at {target}")
            self._downloaded_dir = self._get_cached_download_dir()
            return target

        print("Downloading dataset via kagglehub...")
        download_path = kagglehub.dataset_download(self.dataset_path)
        self._downloaded_dir = Path(download_path)
        print(f"Downloaded to: {self._downloaded_dir}")
        csv_files = list(self._downloaded_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV file found in kagglehub download.")

        shutil.copy(csv_files[0], target)
        print(f"✓ Saved to {target}")
        return target

    def _download_via_rest(self) -> Path:
        """
        Резервная загрузка через REST API Kaggle.
        """
        if not self._rest_available:
            raise RuntimeError("REST API is unavailable: no valid API token.")

        print("Downloading dataset via REST API fallback...")
        url = f"{KAGGLE_API_BASE}/datasets/download/" f"{self.owner}/{self.dataset}"
        response = self._session.get(
            url,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
        zip_path = self.output_dir / "dataset.zip"

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.output_dir)
        zip_path.unlink(missing_ok=True)
        csv_files = list(self.output_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV file found in REST API archive.")
        target = self.output_dir / DATASET_FILE
        if csv_files[0] != target:
            shutil.move(csv_files[0], target)

        print(f"✓ Saved to {target}")
        return target

    def fetch_metadata(self) -> dict:
        """Получить метаданные (REST, если доступен, иначе базовая информация)."""
        # Пытаемся через REST (альтернативный эндпоинт metadata)
        if self._rest_available:
            try:
                url = f"{KAGGLE_API_BASE}/datasets/{self.owner}/{self.dataset}/metadata"
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                raw = resp.json()
                return self._parse_metadata(raw)
            except Exception as e:
                logger.warning("REST metadata failed: %s", e)

        # Если REST не сработал – возвращаем базовую информацию
        return {
            "title": f"{self.owner}/{self.dataset}",
            "url": f"https://www.kaggle.com/datasets/{self.owner}/{self.dataset}",
            "note": "Полные метаданные временно недоступны. Откройте URL для подробностей.",
        }

    def list_dataset_files(self) -> list[dict]:
        """Список файлов из локального кэша (работает всегда)."""
        if not self._downloaded_dir:
            self._downloaded_dir = self._get_cached_download_dir()
            if not self._downloaded_dir:
                self.download()

        files = []
        for f in self._downloaded_dir.iterdir():
            if f.is_file():
                files.append(
                    {
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                    }
                )
        return files

    def print_dataset_info(self):
        print("=" * 60)
        print("KAGGLE DATASET INFO")
        print("=" * 60)

        meta = self.fetch_metadata()
        print(f"Title         : {meta.get('title', '')}")
        if "creator" in meta:
            print(f"Creator       : {meta['creator']}")
        print(f"URL           : {meta.get('url', '')}")
        if "total_bytes" in meta:
            print(f"Total size    : {meta['total_bytes'] / (1024**2):.1f} MB")
        if "licenses" in meta:
            print(f"License(s)    : {', '.join(meta['licenses'])}")
        if "tags" in meta:
            print(f"Tags          : {', '.join(meta['tags'])}")
        if "note" in meta:
            print(f"Note          : {meta['note']}")

        print("-" * 60)
        files = self.list_dataset_files()
        print("Files in dataset:")
        for f in files:
            size_kb = f["size_bytes"] / 1024
            print(f"  {f['name']:<40} {size_kb:>10.1f} KB")
        print("=" * 60)

    # ── Вспомогательные методы ─────────────────────────────────────────────
    def _get_cached_download_dir(self):
        cache_base = Path.home() / ".cache" / "kagglehub" / "datasets"
        dataset_dir = cache_base / self.owner / self.dataset / "versions"
        if dataset_dir.exists():
            versions = sorted(dataset_dir.iterdir(), reverse=True)
            for ver in versions:
                if (ver / DATASET_FILE).exists():
                    return ver
        return None

    def _load_credentials_from_json(self):
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            with open(kaggle_json) as fh:
                creds = json.load(fh)
            self._username = creds.get("username")
            self._key = creds.get("key")

    def _parse_metadata(self, raw: dict) -> dict:
        """Парсит JSON-ответ от REST API."""
        return {
            "title": raw.get("title", ""),
            "subtitle": raw.get("subtitle", ""),
            "total_bytes": raw.get("totalBytes", 0),
            "usability_rating": raw.get("usabilityRating", 0.0),
            "licenses": [lic.get("name", "") for lic in raw.get("licenses", [])],
            "tags": [tag.get("name", "") for tag in raw.get("tags", [])],
            "creator": raw.get("creatorName", ""),
            "last_updated": raw.get("lastUpdated", ""),
            "download_count": raw.get("downloadCount", 0),
            "vote_count": raw.get("voteCount", 0),
            "url": f"https://www.kaggle.com/datasets/{self.owner}/{self.dataset}",
        }
