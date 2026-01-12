import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
import pandas as pd
import re


@dataclass
class LlmLabelingConfig:
    """
    Конфигурация универсального пайплайна разметки.

    Главное:
      - data_path: путь до входного датасета (csv/parquet)
      - text_col: колонка с текстом для LLM
      - bucket/prefix/...: S3
      - helpers_path: где лежат llm_labeling.py, config.yaml, .ml-job-preset.yml.
        Если не указать, возьмётся директория, где лежит этот файл (labeling/).

      - tmp_root: базовая директория для tmp_files_run_{TS}.
        Если None — папка создаётся в текущем каталоге (где запущен ноут).
      - cleanup_tmp: удалять ли tmp_files_run_{TS} после завершения пайплайна.
    """

    data_path: Optional[str] = None
    text_col: str = "query"
    
    bucket: str = "ecom-ml-dolyame-search"
    prefix: str = "labeling_tmp/synthetic"
    endpoint_url: str = "https://s3-msk.tinkoff.ru"
    access_key: str = ""
    secret_key: str = ""
    helpers_path: Optional[str] = None
    tmp_root: Optional[str] = None
    cleanup_tmp: bool = False
    
    chunk_size: int = 5000
    
    flavor: str = "4cpu-64ram"
    region: str = "ix-m4-sm4"
    gpu_num: int = 2
    max_parallel: int = 4
    mlc_project: str = "ecom-ml"

    system_prompt: Optional[str] = None

    as_json: bool = True

    template_name: str = ".ml-job-preset.yml"
    files_to_copy: List[str] = field(
        default_factory=lambda: [
            "llm_labeling.py",
            "config.yaml",
            ".ml-job-preset.yml",
            "system_prompt.py",
        ]
    )


class LlmLabelingPipeline:
    """
      читает полный датасет,
      выделяет уникальные тексты для разметки,
      режет их на чанки и заливает в S3,
      готовит run-директории,
      запускает mlc jobы,
      скачивает результаты (text_col, llm_raw, llm_json),
      джойнит их обратно к исходному датасету.
    +временная папка: tmp_files_run_{TS} в каталоге запуска или в tmp_root.
    """

    def __init__(self, cfg: LlmLabelingConfig, input_df: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        self._input_df = input_df

        if cfg.helpers_path is None:
            self.helpers_path = Path(__file__).resolve().parent
        else:
            self.helpers_path = Path(cfg.helpers_path).resolve()

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if cfg.tmp_root is None:
            self.tmp_root = Path.cwd() / f"tmp_files_run_{run_ts}"
        else:
            self.tmp_root = Path(cfg.tmp_root).resolve() / f"tmp_files_run_{run_ts}"

        self.data_root = self.tmp_root / "data"
        self.runs_root = self.tmp_root / "runs"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

        self.s3 = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
        )

        self.original_df: Optional[pd.DataFrame] = None 
        self.df: Optional[pd.DataFrame] = None
        self.chunk_keys: List[int] = []

        print(f"[INIT] tmp_root = {self.tmp_root}")
        print(f"[INIT] helpers_path = {self.helpers_path}")

    def run(
        self,
        save_merged: bool = True,
        cleanup_tmp: Optional[bool] = None,
    ) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
        """
        полный цикл пайплайна
        save_merged: сохранить/не сохранить итоговый parquet на диск и вернуть путь + df.
        """
        if cleanup_tmp is None:
            cleanup_tmp = self.cfg.cleanup_tmp

        self._update_system_prompt_file()

        self._load_data()
        self._prepare_chunks_and_upload()
        self._prepare_run_dirs()
        self._submit_jobs()
        out_path, df = self._download_and_merge_results(save_merged=save_merged)

        if cleanup_tmp and self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
            print(f"[CLEANUP] removed tmp_root {self.tmp_root}")

        return out_path, df

    def _update_system_prompt_file(self) -> None:
        prompt = self.cfg.system_prompt
        if not prompt:
            print("[PROMPT] cfg.system_prompt is empty, skip updating system_prompt.py")
            return

        system_prompt_path = self.helpers_path / "system_prompt.py"
        system_prompt_path.parent.mkdir(parents=True, exist_ok=True)

        code = f"SYSTEM_PROMPT = {prompt!r}\n"
        system_prompt_path.write_text(code, encoding="utf-8")
        print(f"[PROMPT] updated SYSTEM_PROMPT in {system_prompt_path}")

    def _patch_config_yaml(self, run_dir: Path) -> None:
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            print(f"[RUN_DIRS] WARN: {cfg_path} not found, skip patch")
            return
    
        txt = cfg_path.read_text(encoding="utf-8")
    
        line = f"as_json: {str(self.cfg.as_json).lower()}"
    
        if re.search(r"(?m)^\s*as_json\s*:", txt):
            txt = re.sub(r"(?m)^\s*as_json\s*:\s*.*$", line, txt)
        else:
            if not txt.endswith("\n"):
                txt += "\n"
            txt += line + "\n"
    
        cfg_path.write_text(txt, encoding="utf-8")
        print(f"[RUN_DIRS] patched {cfg_path} with: {line}")


    def _load_data(self):
        cfg = self.cfg
        if self._input_df is not None:
            full_df = self._input_df.copy()
            print("[LOAD] using in-memory DataFrame")
        else:
            path = cfg.data_path
            if path is None:
                raise ValueError(
                    "Нужно либо указать data_path в конфиге, "
                    "либо передать input_df в LlmLabelingPipeline"
                )
            print(f"[LOAD] {path}")

            if path.endswith(".parquet"):
                full_df = pd.read_parquet(path)
            else:
                full_df = pd.read_csv(path)

        assert cfg.text_col in full_df.columns, f"Нет колонки text_col='{cfg.text_col}'"

        self.original_df = full_df.copy()

        print(f"[LOAD] shape raw = {full_df.shape}")
        dedup_df = full_df.drop_duplicates(subset=[cfg.text_col]).copy()
        print(f"[LOAD] shape after dedup[{cfg.text_col}] = {dedup_df.shape}")

        llm_df = dedup_df[[cfg.text_col]].copy()
        llm_df = llm_df.sort_values(cfg.text_col).reset_index(drop=True)

        self.df = llm_df
        n = len(llm_df)
        num_chunks = math.ceil(n / cfg.chunk_size)
        self.chunk_keys = list(range(num_chunks))
        print(f"[LOAD] N unique texts = {n}, chunk_size = {cfg.chunk_size}, num_chunks = {num_chunks}")

    def _prepare_chunks_and_upload(self):
        assert self.df is not None, "df ещё не загружен"
        cfg = self.cfg
        df = self.df

        print(f"[CHUNKS] writing & uploading {len(self.chunk_keys)} chunks...")
        for i, key in enumerate(self.chunk_keys):
            start = i * cfg.chunk_size
            end = (i + 1) * cfg.chunk_size
            chunk = df.iloc[start:end].copy()

            run_dir = self.data_root / f"data_{key}"
            run_dir.mkdir(parents=True, exist_ok=True)

            chunk = chunk[[cfg.text_col]].drop_duplicates(subset=[cfg.text_col])
            local_csv = run_dir / f"data_for_labeling_{key}_dedup.csv"
            chunk.to_csv(local_csv, index=False)
            print(f"[CHUNKS] [chunk {key}] {len(chunk)} rows -> {local_csv}")

            s3_key = f"{cfg.prefix}/data_for_labeling_{key}_dedup.csv"
            with open(local_csv, "rb") as f:
                body = f.read()
            self.s3.put_object(Bucket=cfg.bucket, Key=s3_key, Body=body)
            print(f"[UPLOAD] s3://{cfg.bucket}/{s3_key}")

    def _prepare_run_dirs(self):
        cfg = self.cfg
        template_path = self.helpers_path / cfg.template_name

        if not template_path.exists():
            raise FileNotFoundError(f"Не найден шаблон: {template_path}")

        template_text = template_path.read_text(encoding="utf-8")

        print(f"[RUN_DIRS] preparing in {self.runs_root}")
        for key in self.chunk_keys:
            run_dir = self.runs_root / str(key)
            run_dir.mkdir(exist_ok=True)
            print(f"[RUN_DIRS] +folder: {run_dir}")

            for fname in cfg.files_to_copy:
                src = self.helpers_path / fname
                if not src.exists():
                    print(f"[RUN_DIRS] WARN: {src} not found, skip")
                    continue
                dst = run_dir / src.name
                dst.write_bytes(src.read_bytes())
                print(f"[RUN_DIRS] copy {src} -> {dst}")

            self._patch_config_yaml(run_dir)

            rendered = (
                template_text.replace("{key}", str(key))
                .replace("{flavor}", cfg.flavor)
                .replace("{region}", cfg.region)
                .replace("{PREFIX}", cfg.prefix)
                .replace("{access_key}", cfg.access_key)
                .replace("{secret_key}", cfg.secret_key)
                .replace("{gpu_num}", str(cfg.gpu_num))
            )

            out_preset = run_dir / ".ml-job-preset.yml"
            out_preset.write_text(rendered, encoding="utf-8")
            print(f"[RUN_DIRS] preset: {out_preset}")

    def _run_mlc_job(self, key: int) -> Tuple[int, int, str, str]:
        cfg = self.cfg
        run_dir = self.runs_root / str(key)
        if not run_dir.is_dir():
            return key, -1, "", f"[ERR] нет папки {run_dir}"
    
        print(f"[MLC] start key={key}")
    
        env = os.environ.copy()
        env["HOME"] = "/tmp/emptyhome"
        env["XDG_CONFIG_HOME"] = "/tmp/emptyhome"
        os.makedirs(env["HOME"], exist_ok=True)
    
        result = subprocess.run(
            ["mlc", "job", "submit", "-p", cfg.mlc_project],
            cwd=run_dir,
            text=True,
            capture_output=True,
            env=env,
        )
    
        print(f"[MLC] done  key={key}, code={result.returncode}")
        return key, result.returncode, result.stdout, result.stderr


    def _submit_jobs(self):
        cfg = self.cfg
        results = []
        print(f"[MLC] submitting {len(self.chunk_keys)} jobs, max_parallel={cfg.max_parallel}")

        with ThreadPoolExecutor(max_workers=cfg.max_parallel) as executor:
            fut2key = {executor.submit(self._run_mlc_job, k): k for k in self.chunk_keys}
            for fut in as_completed(fut2key):
                key, code, out, err = fut.result()
                results.append((key, code))
                if code != 0:
                    print(f"\n[MLC][ERROR] key={key}, code={code}")
                    if err:
                        print("--- stderr ---")
                        print(err[:10000], "...")
                else:
                    if out.strip():
                        print(f"\n[MLC][stdout key={key}]")
                        print(out[:400], "...")

        print("\n[MLC] summary:")
        for key, code in sorted(results):
            print(f"  key={key}: code={code}")

    @staticmethod
    def _list_parquets(bucket: str, prefix: str, s3_client) -> List[dict]:
        keys = []
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = s3_client.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                if o["Key"].endswith(".parquet"):
                    keys.append(
                        {
                            "Key": o["Key"],
                            "Size": o["Size"],
                            "LastModified": o["LastModified"],
                        }
                    )
            token = resp.get("NextContinuationToken")
            if not token:
                break
        return sorted(keys, key=lambda x: x["LastModified"], reverse=True)

    def _download_and_merge_results(
        self,
        save_merged: bool = True,
    ) -> Tuple[Optional[Path], Optional[pd.DataFrame]]:
        cfg = self.cfg
        downloaded_paths: List[Path] = []

        print("[RESULT] downloading outputs...")
        for key in self.chunk_keys:
            prefix_out = f"{cfg.prefix}/output_{key}/"
            files = self._list_parquets(cfg.bucket, prefix_out, self.s3)
            print(f"\n[RESULT] key={key}: {len(files)} parquet(s)")
            for f in files[:5]:
                print(f'  {f["LastModified"].isoformat()}  {f["Size"]}  s3://{cfg.bucket}/{f["Key"]}')

            if not files:
                continue

            local_dir = self.data_root / f"data_{key}"
            local_dir.mkdir(parents=True, exist_ok=True)

            latest = files[0]
            local_path = local_dir / Path(latest["Key"]).name
            self.s3.download_file(cfg.bucket, latest["Key"], str(local_path))
            print(f"[RESULT] downloaded latest for key={key} -> {local_path}")
            downloaded_paths.append(local_path)

        if not downloaded_paths:
            print("[RESULT] нет скачанных parquet, нечего склеивать")
            return None, None

        llm_dfs = [pd.read_parquet(p) for p in downloaded_paths]
        llm_df = pd.concat(llm_dfs, ignore_index=True)

        if self.original_df is None:
            raise RuntimeError("original_df is None, _load_data не был вызван")

        text_col = cfg.text_col
        llm_df = llm_df.drop_duplicates(subset=[text_col], keep="last")
        merged = self.original_df.merge(llm_df, on=text_col, how="left")

        out_path: Optional[Path] = None
        if save_merged:
            out_path = self.tmp_root / "labeled_all.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(out_path, index=False)
            print(f"\n[RESULT] merged parquet saved to: {out_path} (shape={merged.shape})")
        else:
            print(f"\n[RESULT] merged in memory only (shape={merged.shape}), not saved to disk")

        return out_path, merged
