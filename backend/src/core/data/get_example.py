from backend.src.core.data.preprocessing import GlossDataSource

def text_from_repo(path, logger):
    logger.info("Обработка входных данных")
    data_source = GlossDataSource(path)
    gloss_entries = data_source.get_gloss_entries()
    logger.info('Данные получены')
    return gloss_entries
