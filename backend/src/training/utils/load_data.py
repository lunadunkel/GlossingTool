import logging
import os
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
from typing import Optional, Tuple, List

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

backend_str = str(BACKEND_DIR)
if backend_str not in sys.path:
    sys.path.insert(0, backend_str)

from sklearn.model_selection import train_test_split
from src.core.data.project_exceptions import DataIsMissing
from src.core.config import DataConfig
from src.core.data.preprocessing import GlossDataSource

def clone_and_load_data(config: 'DataConfig', logger: logging.Logger, clone: bool = True, 
                        data_path: Optional[str] = None, inference: bool = False) -> Tuple[List, List, List]:
    
    texts_list = config.texts_list
    train_split=config.train_split
    val_split=config.val_split
    test_split=config.test_split
    random_state=config.random_seed

    if clone:
        load_dotenv()
        logger.info("Клонирование репозитория")
        token = os.getenv("GITHUB_TOKEN")
        user = os.getenv("USER")
        repo = os.getenv("REPO")
        repo_user = os.getenv("REPO_USER")
        if not token or not user or not repo or not repo_user:
            logger.error("GITHUB_TOKEN и USER не были предоставлены")
            raise ValueError("Требуются GITHUB_TOKEN, USER, REPO_USER и REPO")

        repo_path = f"../{repo}"
        if not os.path.exists(repo_path):
            url = f"https://{user}:{token}@github.com/{repo_user}/{repo}.git"
            subprocess.run(["git", "clone", url, repo_path], check=True)

            logger.info("Репозиторий успешно клонирован")
        else:
            logger.info("Репозиторий уже существует")

        logger.info("Обработка входных данных")
        data_source = GlossDataSource(repo_path, texts_list)
    else:
        if data_path is None:
            raise DataIsMissing
        data_source = GlossDataSource(data_path)
    try:
        gloss_entries = data_source.get_gloss_entries()
        if not gloss_entries:
            raise DataIsMissing
    except Exception as e:
        logger.error(f'В ходе работы программы возникла ошибка: {str(e)}')
        raise Exception(e)
    logger.info("Данные успешно обработаны")

    logger.info("Разбиение данных на train/val/test...")
    train_gloss, temp = train_test_split(gloss_entries, test_size=val_split+test_split, random_state=random_state, shuffle=True)
    val_gloss, test_gloss = train_test_split(temp, test_size=test_split/(val_split+test_split), random_state=random_state, shuffle=True)
    return train_gloss, val_gloss, test_gloss