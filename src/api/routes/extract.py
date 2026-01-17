import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from extract_menu import extract_menu_async

router = APIRouter()


@router.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("balanced"),
    max_pages: Optional[int] = Form(None),
):
    suffix = Path(file.filename or "").suffix
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        data = await extract_menu_async(tmp_path, mode=mode, max_pages=max_pages)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
