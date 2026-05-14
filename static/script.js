// ===== Sample Emails =====
const SAMPLES = {
    spam1: "URGENT! You have won a $1,000,000 prize! Click here NOW to claim your reward. Limited time offer, act fast! Call 1-800-FREE-MONEY immediately!",
    spam2: "Chúc mừng bạn đã trúng thưởng 500 triệu đồng! Nhấn vào link ngay để nhận thưởng. Cơ hội có hạn, hãy nhanh tay đăng ký ngay hôm nay!",
    ham1: "Hi team, please find attached the meeting notes from today's discussion. Let me know if you have any questions about the project timeline. Best regards.",
    ham2: "Chào bạn, cuộc họp nhóm ngày mai sẽ bắt đầu lúc 9h sáng tại phòng họp A3. Nhớ chuẩn bị bài thuyết trình và tài liệu liên quan nhé. Cảm ơn!"
};

// ===== History =====
let history = [];

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    setupTextarea();
    loadHistory();
});

// ===== Background Particles =====
function createParticles() {
    const container = document.getElementById('bgParticles');
    const colors = ['#6366f1', '#a855f7', '#ec4899', '#22c55e'];

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');

        const size = Math.random() * 6 + 3;
        const color = colors[Math.floor(Math.random() * colors.length)];

        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.background = color;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animationDuration = `${Math.random() * 15 + 10}s`;
        particle.style.animationDelay = `${Math.random() * 10}s`;

        container.appendChild(particle);
    }
}

// ===== Textarea =====
function setupTextarea() {
    const textarea = document.getElementById('emailInput');
    const counter = document.getElementById('charCount');

    textarea.addEventListener('input', () => {
        counter.textContent = textarea.value.length;
    });

    // Ctrl+Enter to submit
    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            classifyEmail();
        }
    });
}

// ===== Fill Sample =====
function fillSample(key) {
    const textarea = document.getElementById('emailInput');
    textarea.value = SAMPLES[key];
    document.getElementById('charCount').textContent = textarea.value.length;

    // Micro animation on chip
    const chips = document.querySelectorAll('.chip');
    chips.forEach(c => c.style.transform = '');

    textarea.focus();
}

// ===== Clear Input =====
function clearInput() {
    const textarea = document.getElementById('emailInput');
    textarea.value = '';
    document.getElementById('charCount').textContent = '0';
    document.getElementById('resultCard').classList.add('hidden');
    textarea.focus();
}

// ===== Classify Email =====
async function classifyEmail() {
    const textarea = document.getElementById('emailInput');
    const text = textarea.value.trim();

    if (!text) {
        textarea.classList.add('shake');
        setTimeout(() => textarea.classList.remove('shake'), 500);
        textarea.focus();
        return;
    }

    const btn = document.getElementById('classifyBtn');
    const resultCard = document.getElementById('resultCard');
    const loadingOverlay = document.getElementById('loadingOverlay');

    // Show loading
    btn.disabled = true;
    resultCard.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error('Server error');
        }

        const data = await response.json();
        displayResult(data, text);
        addToHistory(data, text);

    } catch (error) {
        console.error('Error:', error);
        alert('Có lỗi xảy ra khi phân loại. Vui lòng thử lại.');
    } finally {
        btn.disabled = false;
        loadingOverlay.classList.add('hidden');
    }
}

// ===== Display Result =====
function displayResult(data, text) {
    const resultCard = document.getElementById('resultCard');
    const resultHeader = document.getElementById('resultHeader');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');

    const isSpam = data.prediction === 'spam';

    // Header styling
    resultHeader.className = `result-header ${isSpam ? 'spam' : 'ham'}`;
    resultIcon.textContent = isSpam ? '🚫' : '✅';
    resultTitle.textContent = isSpam ? 'EMAIL RÁC (SPAM)' : 'EMAIL BÌNH THƯỜNG (HAM)';
    resultSubtitle.textContent = isSpam
        ? 'Email này được phân loại là spam - có khả năng là thư rác hoặc lừa đảo.'
        : 'Email này được phân loại là ham - nội dung bình thường, an toàn.';

    // Probability bars
    const hamProb = data.probabilities.ham || 0;
    const spamProb = data.probabilities.spam || 0;

    document.getElementById('hamProb').textContent = `${hamProb}%`;
    document.getElementById('spamProb').textContent = `${spamProb}%`;

    // Animate bars after a small delay
    setTimeout(() => {
        document.getElementById('hamBar').style.width = `${hamProb}%`;
        document.getElementById('spamBar').style.width = `${spamProb}%`;
    }, 100);

    // Stats
    document.getElementById('tokenCount').textContent = data.tokens_count;
    document.getElementById('vocabMatch').textContent = data.vocab_matches;

    const confidence = Math.max(hamProb, spamProb);
    document.getElementById('confidence').textContent = `${confidence}%`;

    // Show card
    resultCard.classList.remove('hidden');

    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===== History Management =====
function addToHistory(data, text) {
    const entry = {
        text: text.substring(0, 120),
        prediction: data.prediction,
        confidence: Math.max(data.probabilities.ham || 0, data.probabilities.spam || 0),
        time: new Date().toLocaleTimeString('vi-VN')
    };

    history.unshift(entry);
    if (history.length > 20) history.pop();

    saveHistory();
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('historyList');

    if (history.length === 0) {
        list.innerHTML = `
            <div class="history-empty">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none" opacity="0.3">
                    <circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="2" fill="none"/>
                    <path d="M24 14V24L30 30" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p>Chưa có lịch sử phân loại</p>
            </div>
        `;
        return;
    }

    list.innerHTML = history.map((entry, i) => `
        <div class="history-item" style="animation-delay: ${i * 0.05}s">
            <span class="history-badge ${entry.prediction}">${entry.prediction.toUpperCase()}</span>
            <div class="history-content">
                <div class="history-text">${escapeHtml(entry.text)}${entry.text.length >= 120 ? '...' : ''}</div>
                <div class="history-meta">${entry.time} &bull; Độ tin cậy: ${entry.confidence}%</div>
            </div>
        </div>
    `).join('');
}

function clearHistory() {
    history = [];
    saveHistory();
    renderHistory();
}

function saveHistory() {
    try {
        localStorage.setItem('spam_history', JSON.stringify(history));
    } catch (e) {}
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('spam_history');
        if (saved) {
            history = JSON.parse(saved);
            renderHistory();
        }
    } catch (e) {}
}

// ===== Utility =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
