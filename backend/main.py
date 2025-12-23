import json
import os
import re
import subprocess
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from fastapi.responses import FileResponse, JSONResponse
from backend.src.core.data.get_example import text_from_repo
from backend.src.core.lemma_inserter import LemmaInserter
from backend.src.training.utils.logger import setup_logger

app = FastAPI(title="Glossing API")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

BACKEND_DIR = Path(__file__).parent 
LOG_DIR = str(BACKEND_DIR / "logs")
LOGGER = setup_logger('fastAPI', log_dir=LOG_DIR)

DEMO_SENTENCES = text_from_repo('backend/src/core/data/demo.txt', LOGGER)
PROJECT_ROOT = Path(__file__).parent.parent

@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response

class Sentence(BaseModel):
    id: int
    text: str

class SegmentationRequest(BaseModel):
    id: int
    text: str

class SegmentationResponse(BaseModel):
    segments: list

class GlossRequest(BaseModel):
    segmentation: list

class GlossResponse(BaseModel):
    segmentation: list
    glosses: list
    

@app.get("/")
def read_root():
    """Корневой эндпоинт"""
    LOGGER.info('Основная страница открыта')
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/sentences", response_model=List[Sentence])
def get_sentences():
    """Получить все предложения из репозитория"""
    LOGGER.info('Получен get запрос на выбор предложения из репозитория')
    original_sentences = [{'id': sent.id, 'text': sent.segmented} for sent in DEMO_SENTENCES]
    return original_sentences


@app.get("/sentences/{sentence_id}", response_model=Sentence)
def get_sentence(sentence_id: int):
    LOGGER.info(f'Было выбрано предложение номер {sentence_id}')
    if len(DEMO_SENTENCES) > sentence_id:
        return {'id': sentence_id, 'text': DEMO_SENTENCES[sentence_id].segmented}
    LOGGER.error('Предложение не найдено, status_code=404')
    raise HTTPException(status_code=404, detail="Предложение не найдено")


@app.post("/segment")
def segment_text(request: SegmentationRequest, response_model=SegmentationResponse):
    """Автоматическая сегментация текста на морфемы."""
    text = request.text.strip()
    LOGGER.info(f'Запрос на сегментацию текста: {text}')
    if not text:
        LOGGER.error('Текст не предоставлен, status_code=404')
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    os.makedirs('backend/src/core/data/temp', exist_ok=True)
    LOGGER.info(f'Запись данных в temp')
    with open('backend/src/core/data/temp/temp.txt', 'w') as file:
        file.write(f'1>\t{text}')

    LOGGER.info(f'Запрос на inference')
    result = subprocess.run([sys.executable,
        "src/inference/inference_script.py",
        "--config", "yaml_configs/CNN.yaml",
        "--data_path", "src/core/data/temp/temp.txt",
        "--device", "cpu",
        "--name", 'fastAPI',
        "--logs", LOG_DIR],
        cwd=BACKEND_DIR
    )

    if result.returncode != 0:
        LOGGER.error("Ошибка в subprocess:")
        LOGGER.error("STDOUT:", result.stdout)
        LOGGER.error("STDERR:", result.stderr)
        raise RuntimeError("Инференс завершился с ошибкой")

    with open('backend/src/core/data/temp/temp_segmentation.txt') as file:
        segments = []
        for line in file:
            if segment := re.match(r'(\d+\>\t)(.*)', line.strip()):
                segments.append(segment.group(2))
        final_segments = []
        for seg in segments:
            final_segments.extend(seg.split('\t'))
    LOGGER.info('Response отправлен')
    return SegmentationResponse(segments=final_segments)


@app.post("/gloss")
def create_gloss(request: GlossRequest, response_model=GlossResponse):
    """Для предложения возвращает список всех возможных глосс"""
    glosses = []
    text = '\t'.join(request.segmentation)
    LOGGER.info(f'Запрос на inference подстановки глосс')
    result = subprocess.run(
        [sys.executable, "src/core/lemma_inserter.py", "--sent", text],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        LOGGER.error("Ошибка в subprocess:")
        LOGGER.error("STDOUT:", result.stdout)
        LOGGER.error("STDERR:", result.stderr)
        raise RuntimeError("Инференс завершился с ошибкой")

    try:
        LOGGER.info(f'Запись в temp')
        with open('backend/src/core/data/temp/temp_glossing.json') as file:
            glosses = json.load(file)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ошибка в обработке json: {result.stdout}") from e
    LOGGER.info(f'Response отправлен')
    return GlossResponse(segmentation=request.segmentation, glosses=glosses)


@app.get("/health")
def health_check():
    """Проверка работоспособности API"""
    return {"status": "healthy"}

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")