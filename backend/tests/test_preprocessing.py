"""
Юнит-тесты для модуля preprocessing
"""
import pytest
from src.core.data.preprocessing import (
    GlossingPreprocessor, Token, TokenType, GlossEntry
)


@pytest.fixture
def preprocessor():
    """Фикстура для создания экземпляра препроцессора"""
    return GlossingPreprocessor()


class TestGlossingPreprocessor:
    """Тесты для класса GlossingPreprocessor"""
    
    def test_tokenize_simple_line(self, preprocessor):
        """Тест токенизации простой строки"""
        line = "слово второе третье"
        tokens = preprocessor._tokenize(line)
        assert len(tokens) == 3
        assert tokens == ["слово", "второе", "третье"]
    
    def test_tokenize_empty_line(self, preprocessor):
        """Тест токенизации пустой строки"""
        line = ""
        tokens = preprocessor._tokenize(line)
        assert tokens == []
    
    def test_tokenize_single_word(self, preprocessor):
        """Тест токенизации одного слова"""
        line = "слово"
        tokens = preprocessor._tokenize(line)
        assert len(tokens) == 1
        assert tokens[0] == "слово"
    
    def test_tokenize_extra_spaces(self, preprocessor):
        """Тест токенизации строки с лишними пробелами"""
        line = "  слово   через  много пробелов     "
        tokens = preprocessor._tokenize(line)
        # После strip и split не должно остаться пустых токенов
        assert all(token for token in tokens)
    
    def test_classify_token_word(self, preprocessor):
        """Тест классификации обычного слова"""
        token = preprocessor._classify_token("слово")
        assert isinstance(token, Token)
        assert token.text == "слово"
        assert token.type == TokenType.WORD
        assert token.mask_for_bio == False
    
    def test_classify_token_speaker(self, preprocessor):
        """Тест классификации токена спикера"""
        token = preprocessor._classify_token("[ГД:]")
        assert token.type == TokenType.SPEAKER
        assert token.mask_for_bio == True
    
    def test_classify_token_with_bracketed_flag(self, preprocessor):
        """Тест классификации с флагом bracketed"""
        token = preprocessor._classify_token("а вот так", bracketed=True)
        assert token.type == TokenType.BRACKETED
        assert token.mask_for_bio == True
    
    def test_find_bracketed_tokens_simple(self, preprocessor):
        """Тест поиска токенов в скобках - простой случай"""
        tokens = ["обычное", "(уберите)", "опять"]
        result = preprocessor._find_bracketed_tokens(tokens)
        assert len(result) == 3
        assert result == [False, True, False]
    
    def test_find_bracketed_tokens_multiword(self, preprocessor):
        """Тест поиска токенов в скобках - многословное выражение"""
        tokens = ["слово", "(а", "тут", "много)", "вернулись"]
        result = preprocessor._find_bracketed_tokens(tokens)
        assert len(result) == 5
        assert result[0] == False
        assert result[1] == True
        assert result[2] == True
        assert result[3] == True
        assert result[4] == False
    
    def test_find_bracketed_tokens_no_brackets(self, preprocessor):
        """Тест поиска токенов в скобках - нет скобок"""
        tokens = ["здесь", "все", "норм"]
        result = preprocessor._find_bracketed_tokens(tokens)
        assert all(not b for b in result)
    
    def test_find_bracketed_tokens_with_punctuation(self, preprocessor):
        """Тест обработки скобок с пунктуацией"""
        tokens = ["если,", "(вот так),", "захотели."]
        result = preprocessor._find_bracketed_tokens(tokens)
        # Пунктуация удаляется перед проверкой
        assert len(result) == 3
    
    def test_process_line(self, preprocessor):
        """Тест полной обработки строки"""
        line = "работай (как-нибудь) [ДС:] нормально"
        tokens = preprocessor.process_line(line)
        
        assert len(tokens) == 4
        assert all(isinstance(t, Token) for t in tokens)
        
        assert tokens[0].type == TokenType.WORD
        assert tokens[1].type == TokenType.BRACKETED
        assert tokens[2].type == TokenType.SPEAKER
        assert tokens[3].type == TokenType.WORD
    
    def test_process_line_empty(self, preprocessor):
        """Тест обработки пустой строки"""
        line = ""
        tokens = preprocessor.process_line(line)
        assert tokens == []


class TestToken:
    """Тесты для класса Token"""
    
    def test_token_creation(self):
        """Тест создания токена"""
        token = Token(text="слово", type=TokenType.WORD, mask_for_bio=False)
        assert token.text == "слово"
        assert token.type == TokenType.WORD
        assert token.mask_for_bio == False
    
    def test_token_repr(self):
        """Тест строкового представления токена"""
        token = Token(text="слово", type=TokenType.WORD, mask_for_bio=False)
        repr_str = repr(token)
        assert "слово" in repr_str
        assert "WORD" in repr_str
        assert "False" in repr_str


class TestTokenType:
    """Тесты для enum TokenType"""
    
    def test_token_types_exist(self):
        """Тест наличия всех типов токенов"""
        assert TokenType.WORD.value == "word"
        assert TokenType.SPEAKER.value == "speaker"
        assert TokenType.BRACKETED.value == "bracketed"
    
    def test_token_types_count(self):
        """Тест количества типов токенов"""
        types = list(TokenType)
        assert len(types) == 3


class TestGlossEntry:
    """Тесты для класс GlossEntry"""
    
    def test_gloss_entry_creation(self):
        """Тест создания GlossEntry"""
        entry = GlossEntry(
            id=0,
            segmented="слово-морфема",
            glossed="слово-GLOSS",
            translation="проверочка небольшая",
            metadata="все ради тестов",
            text_name="test.txt",
            line_num="15"
        )
        assert entry.id == 0
        assert entry.segmented == "слово-морфема"
        assert entry.glossed == "слово-GLOSS"
        assert entry.translation == "проверочка небольшая"
    
    def test_gloss_entry_post_init(self):
        """Тест __post_init__ для GlossEntry"""
        entry = GlossEntry(
            id=0,
            segmented="слово-морфема другое-слово",
            glossed="слово-GLOSS другое.слово-слово",
            translation=None,
            metadata=None,
            text_name="test.txt",
            line_num="12"
        )
        # __post_init__ должен создать токены
        assert hasattr(entry, 'orig_tokens')
        assert hasattr(entry, 'segm_tokens')
        assert hasattr(entry, 'gloss_tokens')
        
        assert isinstance(entry.orig_tokens, list)
        assert isinstance(entry.segm_tokens, list)
        assert isinstance(entry.gloss_tokens, list)
    
    def test_gloss_entry_with_optional_fields(self):
        """Тест GlossEntry с опциональными полями"""
        entry = GlossEntry(
            id=1,
            segmented="сегодня",
            glossed="HOY",
            translation=None,
            metadata=None,
            text_name="test.txt",
            line_num="2"
        )
        assert entry.translation is None
        assert entry.metadata is None


class TestPreprocessorCornerCases:
    """Тесты пограничных случаев для препроцессора"""
    
    def test_nested_brackets(self, preprocessor):
        """Тест вложенных скобок"""
        tokens = ["слово", "((какой-то ужас тут произошел))", "другое"]
        result = preprocessor._find_bracketed_tokens(tokens)
        assert len(result) == 3
    
    def test_unclosed_bracket(self, preprocessor):
        """Тест незакрытой скобки"""
        tokens = ["слово", "(так", "вышло"]
        result = preprocessor._find_bracketed_tokens(tokens)
        # Все токены после открывающей скобки должны быть помечены
        assert result[1] == True
        assert result[2] == True
    
    def test_only_closing_bracket(self, preprocessor):
        """Тест только закрывающей скобки"""
        tokens = ["ой", "почему)", "слово"]
        result = preprocessor._find_bracketed_tokens(tokens)
        assert result[1] == True
    
    def test_multiple_punctuation(self, preprocessor):
        """Тест множественной пунктуации"""
        tokens = ["беда...", "пришла,,,", "ужас!!!"]
        result = preprocessor._find_bracketed_tokens(tokens)
        assert len(result) == 3
    
    def test_speaker_variations(self, preprocessor):
        """Тест различных вариантов спикера"""
        variations = [
            "[кто]",
            "[говорит]",
            "[К]",
            "[ВХ]"
        ]
        for var in variations:
            token = preprocessor._classify_token(var)
            assert token.type == TokenType.SPEAKER
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
