#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import find_dotenv, load_dotenv

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_DOTENV_PATH = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parents[1] / ".env")
load_dotenv(_DOTENV_PATH)

from datalab_sdk import AsyncDatalabClient, DatalabClient, ConvertOptions
from datalab_sdk.exceptions import (
    DatalabAPIError,
    DatalabTimeoutError,
    DatalabFileError,
    DatalabValidationError,
)


def _strip_citations(obj: Any) -> Any:
    """
    Datalab may return extra keys like *_citations for each field.
    This removes them to keep output clean JSON.
    """
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k.endswith("_citations"):
                continue
            clean[k] = _strip_citations(v)
        return clean
    if isinstance(obj, list):
        return [_strip_citations(x) for x in obj]
    return obj


def _normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def _parse_price(value: Any) -> Optional[float]:
    """
    Handles number or string prices like:
    "١٢٫٥٠", "12.50", "QAR 12", "12,000", "12.00 QR"
    Extracts the first numeric value found.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    # Normalize Arabic digits and decimal separator
    s = s.translate(ARABIC_DIGITS)
    s = s.replace("٫", ".")  # Arabic decimal separator
    s = s.replace(",", "")   # thousands separator

    # Extract first numeric token
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def build_menu_schema() -> Dict[str, Any]:
    """
    JSON Schema describing what we want to extract.
    We keep it strict and focused: Arabic name, English name, numeric price.
    """
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": (
                    "All menu items. Each entry should correspond to one sellable item "
                    "(ignore headers, category titles, footers)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name_ar": {
                            "type": ["string", "null"],
                            "description": (
                                "Item name in Arabic script exactly as shown in the menu. "
                                "Do not translate or transliterate. If Arabic is not present, return null."
                            )
                        },
                        "name_en": {
                            "type": ["string", "null"],
                            "description": (
                                "Item name in English exactly as shown in the menu. "
                                "Do not translate. If English is not present, return null."
                            )
                        },
                        "description_ar": {
                            "type": ["string", "null"],
                            "description": (
                                "Item description in Arabic. If the menu provides a description, use it verbatim. "
                                "If there is no description but there is an image of the item, describe what is shown "
                                "in 1 short sentence. If neither is available, provide a brief generic description "
                                "based on the item name."
                            )
                        },
                        "description_en": {
                            "type": ["string", "null"],
                            "description": (
                                "Item description in English. If the menu provides a description, use it verbatim. "
                                "If there is no description but there is an image of the item, describe what is shown "
                                "in 1 short sentence. If neither is available, provide a brief generic description "
                                "based on the item name."
                            )
                        },
                        "price": {
                            "type": "number",
                            "description": (
                                "Numeric price for the item. Do NOT include currency symbols."
                            )
                        },
                    },
                    "required": ["name_ar", "name_en", "price"],
                },
            }
        },
        "required": ["items"],
    }


async def extract_menu_async(
    file_path: str, mode: str = "balanced", max_pages: Optional[int] = None
) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    schema = build_menu_schema()

    # Datalab Structured Extraction passes schema via page_schema. :contentReference[oaicite:3]{index=3}
    options = ConvertOptions(
        output_format="json",           # keep JSON blocks available if you want citations/debug
        mode=mode,                      # fast | balanced | accurate :contentReference[oaicite:4]{index=4}
        max_pages=max_pages,
        page_schema=json.dumps(schema), # schema is sent as JSON string in examples :contentReference[oaicite:5]{index=5}
    )

    client = AsyncDatalabClient()  # reads DATALAB_API_KEY env var :contentReference[oaicite:6]{index=6}
    result = await client.convert(file_path, options=options)

    if not getattr(result, "success", False):
        raise RuntimeError(getattr(result, "error", "Conversion failed without an explicit error message."))

    if not getattr(result, "extraction_schema_json", None):
        raise RuntimeError("No extraction_schema_json returned. Check schema or document quality.")

    extracted = json.loads(result.extraction_schema_json)
    extracted = _strip_citations(extracted)

    # Post-cleaning / normalization
    items = extracted.get("items", [])
    cleaned_items: List[Dict[str, Any]] = []

    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue

        name_ar = _normalize_text(it.get("name_ar"))
        name_en = _normalize_text(it.get("name_en"))
        description_ar = _normalize_text(it.get("description_ar"))
        description_en = _normalize_text(it.get("description_en"))
        price = _parse_price(it.get("price"))

        # Skip obviously bad rows
        if not name_ar and not name_en and not description_ar and not description_en and price is None:
            continue

        cleaned_items.append(
            {
                "name_ar": name_ar,
                "name_en": name_en,
                "description_ar": description_ar,
                "description_en": description_en,
                "price": price,
            }
        )

    return {"items": cleaned_items}


def extract_menu(file_path: str, mode: str = "balanced", max_pages: Optional[int] = None) -> Dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(extract_menu_async(file_path, mode=mode, max_pages=max_pages))
    raise RuntimeError("extract_menu is synchronous; use extract_menu_async in async contexts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract menu items (Arabic/English/Price) from PDF or image using Datalab.")
    parser.add_argument("file", help="Path to menu PDF/image (png/jpg/webp/...).")
    parser.add_argument("--out", default=None, help="Output JSON file path. If omitted, prints to stdout.")
    parser.add_argument("--mode", default="accurate", choices=["fast", "balanced", "accurate"], help="Processing mode.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages processed (useful for large PDFs).")
    args = parser.parse_args()

    try:
        data = extract_menu(args.file, mode=args.mode, max_pages=args.max_pages)
    except (DatalabAPIError, DatalabTimeoutError, DatalabFileError, DatalabValidationError) as e:
        raise SystemExit(f"Datalab error: {e}") from e
    except Exception as e:
        raise SystemExit(f"Error: {e}") from e

    out_json = json.dumps(data, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Wrote: {args.out}")
    else:
        print(out_json)


if __name__ == "__main__":
    main()
