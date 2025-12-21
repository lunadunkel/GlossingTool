from src.core.lemma_inserter import LemmaInserter


sent = 'Ӿаӈы    наф     ӿыйк    п’-раф  п’и-р̌   ӿунв-д'

lemma_inserter = LemmaInserter()
lemma_inserter.gloss_sent_lemmas(sent)
