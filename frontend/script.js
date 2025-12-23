let currentStep = 1; // счетчик этапов
let currentMode = 'repo'; // default выбора предложения
let selectedSentenceId = 0; // default выбранное id предложения
let currentText = ''; // актуальный текст
let segments = []; // default сегментация
let glossData = []; // default глоссы
let editingIndex = null; // индекс редактируемого пользователя предложения
let userGlosses = []; // выбранные пользователем глоссы


// вывод предложений из репозитория в html 
function renderSentenceOptions(sentences) {
    const container = document.getElementById('repo-input');
    container.innerHTML = '';
    
    sentences.forEach(function(sentence, index) {
        const option = document.createElement('div');
        option.className = 'sentence-option' + (index === 0 ? ' selected' : '');
        option.setAttribute('data-id', index);
        option.onclick = function() {
            selectSentence(index);
        };
        
        const textDiv = document.createElement('div');
        textDiv.className = 'sentence-text';
        textDiv.textContent = sentence.text;
        option.appendChild(textDiv);
        
        if (sentence.segmentation) {
            const preview = document.createElement('div');
            preview.className = 'sentence-preview';
            preview.textContent = sentence.segmentation.join(' · ');
            option.appendChild(preview);
        }
        
        container.appendChild(option);
    });
}

// подгрузка предложений из бека
async function loadSentencesFromAPI() {
    try {
        const response = await fetch('/sentences');
        if (!response.ok) {
            throw new Error('Ошибка загрузки: ' + response.status);
        }
        const sentence = await response.json();
        console.log('Загружено предложениe:', sentence);
        return sentence;
    } catch (error) {
        console.error('Ошибка при загрузке предложений:', error);
        return [];
    }
}

// подгрузка сегментации из инференса
async function getSegmentation(requestData) {
    try {
        const response = await fetch('/segment', 
                {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(requestData)
                });
        if (!response.ok) {
            throw new Error('Ошибка загрузки: ' + response.status);
        }
        const result = await response.json();
        return result;
    }
    catch (error) {
        console.error('Ошибка:', error);
        return [];
    }
}

// загрузка конкретного предложения 
async function loadCurrSent(id) {
    try {
        const response = await fetch('/sentences/'+id);
        if (!response.ok) {
            throw new Error('Ошибка загрузки: ' + response.status);
        }
        const sentence = await response.json();
        console.log('Загружено предложение:', sentence.text);
        return sentence.text;
    } catch (error) {
        console.error('Ошибка при загрузке предложений:', error);
        return [];
    }
}

// получение глосс
async function getGlossing() {
    try {
        console.log(segments);
        const glossRequest = {
            segmentation: segments
        };
        const response = await fetch('/gloss', {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(glossRequest)
                });
        if (!response.ok) {
            throw new Error('Ошибка загрузки: ' + response.status);
        }
        const GlossData = await response.json();
        return GlossData
    } catch (error) {
        console.error('Ошибка при загрузке предложений:', error);
        return [];
    }
}

loadSentencesFromAPI().then(sentences => {
    console.log('Предложения:', sentences);
    renderSentenceOptions(sentences);
});

// выбранные пользователем глосс
function updateGlossPart(wordIdx, morphIdx, value) {
    if (!userGlosses[wordIdx]) userGlosses[wordIdx] = [];
    userGlosses[wordIdx][morphIdx] = value || '?';
    console.log('Выбрано:', userGlosses);
}

// выбрать режим выбора предложения: из репозитория или вручную
function selectMode(mode) {
    currentMode = mode;
    document.getElementById('mode-manual').classList.remove('selected');
    document.getElementById('mode-repo').classList.remove('selected');
    document.getElementById('mode-' + mode).classList.add('selected');
    
    if (mode === 'manual') {
        document.getElementById('manual-input').style.display = 'block';
        document.getElementById('repo-input').style.display = 'none';
    } else {
        document.getElementById('manual-input').style.display = 'none';
        document.getElementById('repo-input').style.display = 'block';
    }
}

// выбор предложения из репозитория
function selectSentence(id) {
    selectedSentenceId = id;
    const options = document.querySelectorAll('.sentence-option');
    options.forEach(function(el) {
        el.classList.remove('selected');
    });
    document.querySelector('[data-id="' + id + '"]').classList.add('selected');
}

// обработка сегментов
function renderSegments() {
    const container = document.getElementById('segment-container');
    container.innerHTML = '';
    
    segments.forEach(function(seg, idx) {
        const segDiv = document.createElement('div');
        segDiv.className = 'segment';
        
        if (editingIndex === idx) {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'segment-input';
            input.value = seg;
            input.onblur = function() {
                finishEditing(idx, input.value);
            };
            input.onkeypress = function(e) {
                if (e.key === 'Enter') finishEditing(idx, input.value);
            };
            segDiv.appendChild(input);
            container.appendChild(segDiv);
            setTimeout(function() {
                input.focus();
            }, 10);
        } else {
            const span = document.createElement('span');
            span.className = 'segment-text';
            span.textContent = seg;
            span.onclick = function() {
                startEditing(idx);
            };
            segDiv.appendChild(span);
            
            if (idx < segments.length - 1) {
                const mergeBtn = document.createElement('button');
                mergeBtn.className = 'merge-btn';
                mergeBtn.innerHTML = '→';
                mergeBtn.onclick = function(e) {
                    e.stopPropagation();
                    mergeSegments(idx);
                };
                segDiv.appendChild(mergeBtn);
            }
            
            container.appendChild(segDiv);
        }
    });
    console.log(segments);
}

// пользовательское редактирование
function startEditing(idx) {
    editingIndex = idx;
    renderSegments();
}

// обработка введенного пользователем
function finishEditing(idx, newValue) {
    if (newValue.trim()) {
        segments[idx] = newValue.trim();
    }
    editingIndex = null;
    renderSegments();
}

// объединение сегментов
function mergeSegments(idx) {
    segments[idx] = segments[idx] + segments[idx + 1];
    segments.splice(idx + 1, 1);
    
    renderSegments();
}

// переход к этапу step
function goToStep(step) {
    currentStep = step;
    const pages = document.querySelectorAll('.page');
    pages.forEach(function(page) {
        page.classList.remove('active');
    });
    document.getElementById('page' + step).classList.add('active');
    
    for (let i = 1; i <= 4; i++) {
        const circle = document.getElementById('step' + i);
        const line = document.getElementById('line' + i);
        
        if (i < step) {
            circle.classList.add('completed');
            circle.classList.remove('active');
            if (line) line.classList.add('active');
        } else if (i === step) {
            circle.classList.add('active');
            circle.classList.remove('completed');
        } else {
            circle.classList.remove('active', 'completed');
            if (line) line.classList.remove('active');
        }
    }
}

async function goToStep2() {
    segments = [];
    document.getElementById('segment-container').innerHTML = '';

    if (currentMode === 'repo') {
        const curr = await loadCurrSent(selectedSentenceId);
        if (!curr) {
            alert('Предложение не найдено');
            return;
        }
        currentText = curr;
    } else {
        currentText = document.getElementById('input-text').value.trim();
        if (!currentText) {
            alert('Пожалуйста, введите текст.');
            return;
        }
    }

    const loadingEl = document.getElementById('segment-loading-step1');
    const nextBtn = document.querySelector('#page1 .btn-primary');
    
    loadingEl.style.display = 'flex';
    nextBtn.disabled = true;

    let segmentationResult = null;

    try {
        if (currentMode === 'repo') {
            const requestData = {id: selectedSentenceId, text: currentText};
            const response = await getSegmentation(requestData);
            if (response && response.segments) {
                segmentationResult = response.segments;
            } else if (Array.isArray(response)) {
                segmentationResult = response;
            } else {
                throw new Error('Некорректный ответ от сервера');
            }
        } else {
            const requestData = {id: 0, text: currentText};
            const response = await getSegmentation(requestData);
            if (response && response.segments) {
                segmentationResult = response.segments;
            } else if (Array.isArray(response)) {
                segmentationResult = response;
            } else {
                throw new Error('Некорректный ответ от сервера');
            }
        }

        segments = segmentationResult;

        document.getElementById('original-text').textContent = currentText;
        renderSegments();

        goToStep(2);

    } catch (error) {
        console.error('Ошибка сегментации:', error);
        alert('Не удалось выполнить сегментацию. Попробуйте снова.');
    } finally {
        loadingEl.style.display = 'none';
        nextBtn.disabled = false;
    }
}

function renderGlossing() {
    const container = document.getElementById('gloss-container');
    container.innerHTML = '';

    glossData.segmentation.forEach((word, wordIdx) => {
        const card = document.createElement('div');
        card.className = 'gloss-card';

        const wordDiv = document.createElement('div');
        wordDiv.className = 'gloss-word';
        wordDiv.textContent = word;
        card.appendChild(wordDiv);

        const morphemes = word.split('-');

        if (morphemes.length > 1) {
            const partsDiv = document.createElement('div');
            partsDiv.className = 'morpheme-parts';
            morphemes.forEach(part => {
                const partSpan = document.createElement('span');
                partSpan.className = 'morpheme-part';
                partSpan.textContent = part;
                partsDiv.appendChild(partSpan);
            });
            card.appendChild(partsDiv);
        }

        const section = document.createElement('div');
        section.className = 'gloss-section';

        const sectionTitle = document.createElement('div');
        sectionTitle.className = 'gloss-section-title';
        sectionTitle.textContent = 'Глоссы для морфем';
        section.appendChild(sectionTitle);

        const fieldsDiv = document.createElement('div');
        fieldsDiv.className = 'gloss-fields';

        morphemes.forEach((morpheme, morphIdx) => {
            const field = document.createElement('div');
            field.className = 'gloss-field';

            const label = document.createElement('label');
            label.className = 'gloss-label';
            label.textContent = `Морфема: ${morpheme}`;
            field.appendChild(label);

            const options = glossData.glosses[wordIdx]?.[morphIdx] || [];

            if (options.length > 0) {
                const select = document.createElement('select');
                select.onchange = function(e) {
                    updateGlossPart(wordIdx, morphIdx, e.target.value);
                };

                const emptyOption = document.createElement('option');
                emptyOption.value = '';
                emptyOption.textContent = '— выберите —';
                select.appendChild(emptyOption);

                options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    select.appendChild(option);
                });

                field.appendChild(select);
            } else {
                const input = document.createElement('input');
                input.type = 'text';
                input.placeholder = 'Введите глоссу...';
                input.onchange = function(e) {
                    updateGlossPart(wordIdx, morphIdx, e.target.value);
                };
                field.appendChild(input);
            }

            fieldsDiv.appendChild(field);
        });

        section.appendChild(fieldsDiv);
        card.appendChild(section);
        container.appendChild(card);
    });
}


async function goToStep3() {
    document.getElementById('segmented-display').textContent = segments.join(' ');
    glossData = await getGlossing();
    renderGlossing();
    goToStep(3);
}

function goToStep4() {
    const original = currentText;
    const segmented = segments.join(' ');

    const glossedWords = segments.map((word, wordIdx) => {
        const morphemes = word.split('-');
        const glossesForWord = morphemes.map((_, morphIdx) => {
            return userGlosses[wordIdx]?.[morphIdx] || '?';
        });
        return glossesForWord.join('-');
    });

    const glossed = glossedWords.join(' ');

    document.getElementById('result-original').textContent = original;
    document.getElementById('result-segmented').textContent = segmented;
    document.getElementById('result-glossed').textContent = glossed;

    const translationEl = document.getElementById('result-translation');
    translationEl.textContent = 'Введите перевод предложения...';
    translationEl.dataset.isPlaceholder = 'true';

    translationEl.contentEditable = true;
    translationEl.addEventListener('focus', function() {
        if (this.dataset.isPlaceholder === 'true') {
            this.textContent = '';
            delete this.dataset.isPlaceholder;
        }
    });
    translationEl.addEventListener('blur', function() {
        if (!this.textContent.trim()) {
            this.textContent = 'Введите перевод предложения...';
            this.dataset.isPlaceholder = 'true';
        }
    });

    goToStep(4);
}

function copyToClipboard(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(function() {
        const btn = event.target;
        const original = btn.textContent;
        setTimeout(function() {
            btn.textContent = original;
        }, 1500);
    });
}

function copyAll() {
    const original = document.getElementById('result-original').textContent;
    const segmented = document.getElementById('result-segmented').textContent;
    const glossed = document.getElementById('result-glossed').textContent;
    let translation = document.getElementById('result-translation').textContent;

    if (translation === 'Введите перевод предложения...' || !translation.trim()) {
        translation = '';
    }

    const lines = [original, segmented, glossed].filter(line => line.trim());
    if (translation) {
        lines.push(`'${translation}'`);
    }

    const fullText = lines.join('\n');

    navigator.clipboard.writeText(fullText).then(() => {
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = 'Скопировано';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Ошибка копирования:', err);
        alert('Не удалось скопировать. Попробуйте вручную.');
    });
}

function reset() {
    goToStep(1);
    currentMode = 'repo';
    selectedSentenceId = 0;
    currentText = '';
    segments = [];
    glossData = null;
    userGlosses = [];
    editingIndex = null;

    document.getElementById('input-text').value = '';
    selectSentence(0);
    selectMode('repo');
}

goToStep(1);
