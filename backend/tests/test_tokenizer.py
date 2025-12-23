"""
Юнит-тесты для модуля токенизации
"""
import pytest
from src.tokenization.char_tokenization import CharTokenizer
from src.core.data.preprocessing import Token, TokenType
from vocabularies.id2char import symbols


@pytest.fixture
def tokenizer():
    """Фикстура для создания экземпляра токенизатора"""
    return CharTokenizer(symbols)

class TestCharTokenizer:
    """Тесты для класса CharTokenizer"""
    
    def test_initialization(self, tokenizer):
        """Тест инициализации токенизатора"""
        assert tokenizer is not None
        assert hasattr(tokenizer, 'id2char')
        assert hasattr(tokenizer, 'char2id')
        assert isinstance(tokenizer.id2char, dict)
        assert isinstance(tokenizer.char2id, dict)
    
    def test_tokenize_simple_word(self, tokenizer):
        """Тест токенизации простого слова"""
        result = tokenizer.tokenize(['слово'])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
    
    def test_tokenize_word_with_hyphen(self, tokenizer):
        """Тест токенизации слова с дефисом"""
        result = tokenizer.tokenize(['слово-морфема-ещё-одна'])
        assert isinstance(result, list)
        assert len(result) == 1
        # Дефис должен быть удален при токенизации
        chars = ''.join(result[0])
        assert '-' not in chars or '<PUNC>' in result[0]
    
    def test_tokenize_multiple_words(self, tokenizer):
        """Тест токенизации нескольких слов"""
        words = ['слово', 'второе', 'третье']
        result = tokenizer.tokenize(words)
        assert len(result) == 3
        assert all(isinstance(chars, list) for chars in result)
    
    def test_tokenize_empty_list(self, tokenizer):
        """Тест токенизации пустого списка"""
        result = tokenizer.tokenize([])
        assert result == []
    
    def test_tokenize_with_token_objects(self, tokenizer):
        """Тест токенизации с объектами Token"""
        token = Token(text="нивхское слово", type=TokenType.WORD, mask_for_bio=False)
        result = tokenizer.tokenize([token])
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_tokenize_bracketed_token(self, tokenizer):
        """Тест токенизации токена в скобках"""
        token = Token(text="(лишние слова на русском)", type=TokenType.BRACKETED, mask_for_bio=True)
        result = tokenizer.tokenize([token])
        assert isinstance(result, list)
        # Должен быть заменен на <PUNC>
        assert all(char == '<PUNC>' for char in result[0])
    
    def test_tokenize_speaker_token(self, tokenizer):
        """Тест токенизации токена спикера"""
        token = Token(text="[ВХ:]", type=TokenType.SPEAKER, mask_for_bio=True)
        result = tokenizer.tokenize([token])
        assert isinstance(result, list)
        # Должен быть заменен на <SPEAKER>
        assert all(char == '<SPEAKER>' for char in result[0])
    
    def test_get_tokens_ids_simple(self, tokenizer):
        """Тест получения ID токенов для простого случая"""
        chars = [['c', 'л', 'о', 'в', 'о']]
        ids = tokenizer.get_tokens_ids(chars)
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)
    
    def test_get_tokens_ids_with_separator(self, tokenizer):
        """Тест получения ID с разделителем табуляции"""
        chars = [['c', 'л', 'о', 'в', 'о'], ['c', 'л', 'о', 'в', 'о']]
        ids = tokenizer.get_tokens_ids(chars)
        assert isinstance(ids, list)
        # Должен включать разделитель между словами
        assert len(ids) > len(chars[0]) + len(chars[1])
    
    def test_get_tokens_ids_unknown_char(self, tokenizer):
        """Тест обработки неизвестных символов"""
        chars = [['©', '%', '®']]
        ids = tokenizer.get_tokens_ids(chars)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
    
    def test_long_symbols_priority(self, tokenizer):
        """Тест приоритета длинных символов при токенизации"""
        result = tokenizer.tokenize(['ӄ’оюдь'])
        assert result == [['ӄ’', 'о', 'ю', 'д', 'ь']]


class TestCharTokenizerCornerCases:
    """Тесты corner cases для CharTokenizer"""
    
    def test_empty_string(self, tokenizer):
        """Тест токенизации пустой строки"""
        result = tokenizer.tokenize([''])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == []
    
    def test_only_hyphens(self, tokenizer):
        """Тест строки, состоящей только из дефисов"""
        result = tokenizer.tokenize(['---'])
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_mixed_case(self, tokenizer):
        """Тест обработки смешанного регистра"""
        result = tokenizer.tokenize(['сЛоВо'])
        assert isinstance(result, list)
        assert len(result[0]) == 5
    
    def test_unicode_characters(self, tokenizer):
        """Тест обработки символов"""
        result = tokenizer.tokenize(['тест'])
        assert isinstance(result, list)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
