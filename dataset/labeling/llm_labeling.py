from datetime import datetime
import asyncio
import logging
import os
import time
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.exceptions import RequestException

import hydra
from omegaconf import DictConfig, OmegaConf

import httpx
import httpx_aiohttp
from openai import AsyncOpenAI
import openai

from tqdm.auto import tqdm

from system_prompt import SYSTEM_PROMPT

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
openai.log = "error"


def check_vllm_server(base_url: str = "http://localhost:8000") -> bool:
    """Проверяем, что vLLM-сервер поднят и отвечает по /health."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            logging.info("vLLM server is running and healthy")
            return True
    except RequestException:
        pass

    logging.warning("vLLM server is not accessible yet")
    return False


async def get_chat_completion(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    text_col: str,
    model_name: str,
    retries: int = 5,
    as_json: bool = False, 
) -> Dict[str, Any]:

    last_error: Optional[Exception] = None
    raw_response: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None

    text = item.get(text_col, "")
    if text is None:
        text = ""

    for attempt in range(retries):
        try:
            if as_json:
                user_content = (
                    "TEXT:\n"
                    "```text\n"
                    f"{text}\n"
                    "```\n"
                    "Follow the system prompt. Return STRICTLY one JSON object and nothing else."
                )
            else:
                user_content = (
                    "TEXT:\n"
                    "```text\n"
                    f"{text}\n"
                    "```\n"
                    "Follow the system prompt and return ONLY the final answer as plain text."
                )
    
            completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                model=model_name,
            )
    
            raw_response = completion.choices[0].message.content or ""
    
            if as_json:
                parsed = json.loads(raw_response, strict=False)
                if isinstance(parsed, dict):
                    break
                else:
                    logging.warning(f"Non-dict JSON for text='{text[:50]}...', attempt={attempt+1}")
                    parsed = None
            else:
                parsed = None
                break
    
        except Exception as e:
            last_error = e
            parsed = None
            logging.error(f"Error in get_chat_completion for text='{str(text)[:50]}...': {e}")


    if as_json and not isinstance(parsed, dict):
        parsed = {
            "error": "fallback_parse_error",
            "last_exception": str(last_error) if last_error else None,
        }

    return {
        text_col: text,
        "llm_raw": raw_response,
        "llm_json": parsed,
    }


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))
    logging.info("Finished llm_labeling")


async def async_main(cfg: DictConfig) -> None:
    log_level = getattr(logging, str(cfg.get("logging", "INFO")).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Config:")
    logging.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    input_path: Optional[str] = cfg.get("input_path") or cfg.get("input_data")
    output_stub: str = cfg.get("output_stub") or cfg.get("output_filename", "llm_labels")
    text_col: str = cfg.get("text_col", "query")
    model_name: str = cfg.get("model_name", "llm")
    vllm_base_url: str = cfg.get("vllm_base_url", "http://localhost:8000")
    as_json: bool = bool(cfg.get("as_json", True))
    logging.info(f"as_json={as_json}")

    if not input_path:
        raise ValueError("В конфиге должен быть задан 'input_path' или 'input_data'")

    # === загрузка данных ===
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input_path не найден: {input_path}")

    if input_path.endswith(".parquet"):
        data = pd.read_parquet(input_path)
    else:
        data = pd.read_csv(input_path)

    if text_col not in data.columns:
        raise ValueError(f"В input_data нет колонки '{text_col}'")

    data = data[[text_col]].drop_duplicates(subset=[text_col]).reset_index(drop=True)
    logging.info(f"Loaded {len(data)} unique texts from {input_path}")

    items: List[Dict[str, Any]] = data.to_dict("records")
    logging.info(f"Total items to label: {len(items)}")

    # === ждём vLLM ===
    attempts = 120
    sleep_schedule = [180, 60, 60, 30, 30, 20, 15, 10]

    while not check_vllm_server(vllm_base_url) and attempts > 0:
        if sleep_schedule:
            sleep_time = sleep_schedule.pop(0)
        else:
            sleep_time = 10
        logging.info(f"vLLM not ready, sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        attempts -= 1

    if not check_vllm_server(vllm_base_url):
        logging.error("vLLM сервер не поднялся после нескольких попыток")
        return

    base_url_v1 = vllm_base_url.rstrip("/") + "/v1"

    logging.info("Start processing with vLLM...")
    async with AsyncOpenAI(
        base_url=base_url_v1,
        api_key="EMPTY",
        http_client=httpx_aiohttp.HttpxAiohttpClient(
            timeout=None,
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
            follow_redirects=True,
        ),
        timeout=None,
        max_retries=5,
    ) as client:
        tasks = [
            get_chat_completion(
                client=client,
                item=item,
                text_col=text_col,
                model_name=model_name,
                as_json=as_json,
            )
            for item in items
        ]

        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

    os.makedirs("/work/output", exist_ok=True)

    df_results = pd.DataFrame.from_dict(results)

    output_path = os.path.join(
        "/work/output",
        f'{output_stub}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}_labeled.parquet',
    )
    df_results.to_parquet(output_path, index=False)
    logging.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
