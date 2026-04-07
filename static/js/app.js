// ============================================================
// Project 1a: Crypto Market Risk Intelligence System
// app.js -- Frontend logic: charts, polling, notifications
// ============================================================

const API = "";  // relative URL — works on any host/port (local or Railway)

// ---- STATE ----
let currentRange       = "1H";
let currentPage        = 1;
let showMA             = true;
let showBB             = true;
let showAnomalies      = true;
let currentChartType   = "candlestick";   // "candlestick" | "line"
let lastPrice          = null;
let lastChangePct      = null;
let notifPermission    = false;
let expandedAnomalyId  = null;
let refreshRateMs      = 5000;
let pollingIntervalId  = null;
let activeTab          = "chart";   // default clean view
let strategiesCache    = {};   // keyed by strategy id, populated on load
let showSignals        = true;
let showVolume         = true;
let showForecast       = false;
let currentCoin        = "BTC";
let isDarkMode         = localStorage.getItem("darkMode") === "true";
let showAllSignals     = false;   // when true, anomaly table includes synthetic continuity rows

// Apply dark mode on load
if (isDarkMode) {
    document.documentElement.setAttribute("data-theme", "dark");
}
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.innerHTML = isDarkMode ? '&#9790;' : '&#9728;';
});

// Keep charts full-width when the browser window is resized
window.addEventListener("resize", () => {
    const priceEl  = document.getElementById("price-chart");
    if (priceEl  && priceEl._fullLayout)  Plotly.Plots.resize(priceEl);
});

// ============================================================
// SEEDING OVERLAY
// ============================================================

async function checkSeedingStatus() {
    try {
        const res  = await fetch("/api/status");
        const data = await res.json();
        if (!data.seeding_complete) {
            document.getElementById("seeding-overlay").classList.remove("hidden");
            setTimeout(checkSeedingStatus, 3000);
        } else {
            document.getElementById("seeding-overlay").classList.add("hidden");
            init();
        }
    } catch (e) {
        setTimeout(checkSeedingStatus, 3000);
    }
}

// ============================================================
// INIT
// ============================================================

async function init() {
    requestNotificationPermission();
    await pollLivePrice();
    await loadChart();
    await loadAnomalies();
    await loadContinuitySignals();
    await loadModelStats();
    await loadTradingStrategies();
    await loadTradingSignal();
    await loadPerformanceMetrics();
    startPolling();
}

// ============================================================
// POLLING -- recursive setTimeout so each poll waits for the
// previous one to finish before scheduling the next.
// A visibilitychange listener reschedules immediately when the
// tab comes back into the foreground (browsers throttle/kill
// timers for background tabs, which made polling appear to stop).
// ============================================================

function startPolling() {
    if (pollingIntervalId) clearTimeout(pollingIntervalId);
    scheduleNextPoll();
}

function stopPolling() {
    if (pollingIntervalId) clearTimeout(pollingIntervalId);
    pollingIntervalId = null;
}

async function runPoll() {
    await pollLivePrice();
    if (activeTab === "chart") await loadChart();
    await loadTradingSignal();
    if (activeTab === "anomalies") await loadAnomalies(currentPage);
}

function scheduleNextPoll() {
    pollingIntervalId = setTimeout(async () => {
        try {
            await runPoll();
        } finally {
            // Always reschedule — even if a fetch failed — so polling
            // never silently dies due to a transient network error.
            scheduleNextPoll();
        }
    }, refreshRateMs);
}

async function pollLivePrice() {
    try {
        const res  = await fetch(`${API}/api/price/live?coin=${currentCoin}`);
        const data = await res.json();
        updatePriceHeader(data.price, data.risk);
        renderInsight(data.insight);
        updateApiStatus(data.source);

        // Show the local time of the last successful poll
        const luEl = document.getElementById("last-updated");
        if (luEl) luEl.textContent = new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

        if (data.risk && data.risk.risk_score >= 61) {
            showAnomalyBanner(data.risk);
            triggerNotification(data.risk);
            refreshAnomalyBadge();
        }
    } catch (e) {
        updateApiStatus("offline");
    }
}

async function refreshBTCPrice(event) {
    // Manually refresh BTC price, bypassing cache
    const btn = event.target;
    btn.classList.add("refreshing");
    btn.disabled = true;
    
    try {
        // Force fresh data by adding timestamp parameter
        const res = await fetch(`${API}/api/price/live?force_refresh=true&coin=${currentCoin}`);
        const data = await res.json();
        updatePriceHeader(data.price, data.risk);
        updateApiStatus("refreshed");
        
        // Show brief success feedback
        setTimeout(() => {
            updateApiStatus(data.source);
        }, 2000);
        
        // Reload chart and signal with fresh data
        await loadChart();
        await loadTradingSignal();
    } catch (e) {
        console.error("Manual refresh failed:", e);
        updateApiStatus("error");
    } finally {
        btn.classList.remove("refreshing");
        btn.disabled = false;
    }
}

// ============================================================
// DARK MODE & REFRESH RATE
// ============================================================

function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    localStorage.setItem("darkMode", isDarkMode);
    const btn = document.getElementById('theme-toggle');
    if (isDarkMode) {
        document.documentElement.setAttribute("data-theme", "dark");
        if (btn) btn.innerHTML = '&#9790;';
    } else {
        document.documentElement.removeAttribute("data-theme");
        if (btn) btn.innerHTML = '&#9728;';
    }
}

function updateRefreshRate(value) {
    refreshRateMs = parseInt(value) * 1000;
    stopPolling();
    startPolling();
}

function switchCoin(coin, btn) {
    if (currentCoin === coin) return;
    currentCoin = coin;

    // Update button active states
    document.querySelectorAll(".coin-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");

    // Update header badge and price label
    const COIN_NAMES = { BTC: "Bitcoin", ETH: "Ethereum", SOL: "Solana" };
    const badge = document.getElementById("coin-badge");
    if (badge) badge.textContent = `${COIN_NAMES[coin] || coin} / USD`;
    const priceLabel = document.querySelector(".price-label");
    if (priceLabel) priceLabel.textContent = `${coin}/USD`;

    // Full reload with new coin
    loadChart();
    loadTradingSignal();
    loadAnomalies(1);
    pollLivePrice();
}

// ============================================================
// HEADER UPDATES
// ============================================================

function updatePriceHeader(price, risk) {
    if (!price) return;

    const priceEl  = document.getElementById("live-price");
    const changeEl = document.getElementById("price-change");
    const badgeEl  = document.getElementById("risk-badge");

    priceEl.textContent = `$${Number(price.close).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;

    // Use the true worldwide 24h change from the API (not the per-poll delta)
    if (price.change_24h_pct !== null && price.change_24h_pct !== undefined) {
        const pct24 = price.change_24h_pct;
        changeEl.textContent = `${pct24 >= 0 ? "+" : ""}${pct24.toFixed(2)}%`;
        changeEl.className   = `price-change ${pct24 >= 0 ? "up" : "down"}`;
        lastChangePct = pct24;
    } else if (lastPrice !== null) {
        const diff = price.close - lastPrice;
        const pct  = (diff / lastPrice * 100).toFixed(2);
        changeEl.textContent = `${diff >= 0 ? "+" : ""}${pct}%`;
        changeEl.className   = `price-change ${diff >= 0 ? "up" : "down"}`;
    }
    lastPrice = price.close;

    // Market stats
    const highEl = document.getElementById("stat-high");
    const lowEl  = document.getElementById("stat-low");
    const volEl  = document.getElementById("stat-vol");
    if (highEl) highEl.textContent = `$${Number(price.high).toLocaleString("en-US", { minimumFractionDigits: 0 })}`;
    if (lowEl)  lowEl.textContent  = `$${Number(price.low).toLocaleString("en-US",  { minimumFractionDigits: 0 })}`;
    // Convert USDT volume → coin units (same as chart bars)
    const coinVol = price.close > 0 ? price.volume / price.close : price.volume;
    if (volEl)  volEl.textContent  = formatVolume(coinVol);

    // Risk badge + gauge
    if (risk) {
        const level  = risk.risk_level ? risk.risk_level.toLowerCase() : "low";
        const score  = risk.risk_score || 0;
        const colors = { low: "#22c55e", medium: "#eab308", high: "#ef4444", critical: "#a855f7" };
        const color  = colors[level] || "#22c55e";

        badgeEl.textContent = `${score.toFixed(0)}/100 ${risk.risk_level}`;
        badgeEl.className   = `risk-badge risk-${level}`;

        const fill  = document.getElementById("risk-gauge-fill");
        const label = document.getElementById("risk-gauge-score");
        if (fill)  { fill.style.width = `${score}%`; fill.style.background = color; }
        if (label) { label.textContent = `${score.toFixed(0)} / 100`; label.style.color = color; }
    }
}

function formatVolume(vol) {
    if (!vol) return "--";
    const sym = currentCoin || "BTC";
    if (vol >= 1e6) return `${(vol / 1e6).toFixed(2)}M ${sym}`;
    if (vol >= 1e3) return `${(vol / 1e3).toFixed(2)}K ${sym}`;
    return `${Number(vol).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4})} ${sym}`;
}

function updateApiStatus(source) {
    const el       = document.getElementById("api-status");
    const textEl   = document.getElementById("status-text");
    if (source === "cache") {
        el.className     = "api-status status-cached";
        textEl.textContent = "Using Cache";
    } else if (source === "binance") {
        el.className     = "api-status status-cached";
        textEl.textContent = "Binance";
    } else if (source === "offline") {
        el.className     = "api-status status-offline";
        textEl.textContent = "Offline";
    } else {
        el.className     = "api-status status-live";
        textEl.textContent = "Live";
    }
}

// ============================================================
// TABS
// ============================================================

function switchTab(name, btn) {
    // Fail-safe: hide every tab section first, then reveal exactly one.
    document.querySelectorAll("section[id^='tab-']").forEach(el => {
        el.classList.remove("active");
        el.classList.add("hidden");
        el.style.display = "none";
    });
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

    const targetTab = document.getElementById(`tab-${name}`);
    if (!targetTab) return;
    targetTab.classList.add("active");
    targetTab.classList.remove("hidden");
    targetTab.style.display = "block";
    if (btn) btn.classList.add("active");

    activeTab = name;

    if (name === "anomalies") {
        loadAnomalies();
        loadContinuitySignals();
    }
    if (name === "model-stats")    loadModelStats();
    if (name === "trading-signal") {
        loadTradingStrategies();
        loadTradingSignal();
    }
    if (name === "performance")    loadPerformanceMetrics();
}

function renderInsight(insight) {
    const headlineEl = document.getElementById("risk-insight-headline");
    const detailEl = document.getElementById("risk-insight-detail");
    if (!headlineEl || !detailEl || !insight) return;
    headlineEl.textContent = insight.headline || "Risk insight unavailable.";
    detailEl.textContent = insight.detail || "Awaiting stable live context for summary.";
}

function switchExplanation(expName, btn) {
    document.querySelectorAll(".exp-content").forEach(el => {
        el.classList.remove("active");
        el.classList.add("hidden");
    });
    document.querySelectorAll(".exp-tab-btn").forEach(el => el.classList.remove("active"));

    const expEl = document.getElementById(`exp-${expName}`);
    if (expEl) {
        expEl.classList.remove("hidden");
        expEl.classList.add("active");
    }
    if (btn) btn.classList.add("active");
}

// ============================================================
// TRADING SIGNALS
// ============================================================

async function loadTradingSignal() {
    try {
        // Get selected strategy from localStorage or use default
        const selectedStrategy = localStorage.getItem("selectedTradingStrategy") || "conservative";
        
        const res = await fetch(`${API}/api/trading-signal?strategy=${selectedStrategy}&coin=${currentCoin}`);
        const data = await res.json();
        const signal = data.signal;

        const cardEl = document.getElementById("signal-card");
        const emojiEl = document.getElementById("signal-emoji");
        const typeEl = document.getElementById("signal-type");
        const confEl = document.getElementById("signal-confidence");
        const priceEl = document.getElementById("signal-price");
        const riskEl = document.getElementById("signal-risk");
        const statusEl = document.getElementById("signal-status");
        const reasoningEl = document.getElementById("signal-reasoning");
        const recEl = document.getElementById("signal-recommendation");

        if (cardEl) cardEl.className = `signal-card signal-${signal.signal.toLowerCase()}`;
        if (emojiEl) emojiEl.textContent = signal.emoji || "🟡";
        if (typeEl) typeEl.textContent = signal.signal;
        if (confEl) confEl.textContent = `${signal.confidence.toFixed(0)}%`;
        if (priceEl) priceEl.textContent = `$${Number(data.current_price).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
        if (riskEl) riskEl.textContent = `${data.risk_score.toFixed(1)}/100`;
        if (statusEl) statusEl.textContent = signal.signal;
        if (reasoningEl) reasoningEl.innerHTML = `<p><strong>Reasoning:</strong> ${signal.reasoning}</p>`;
        if (recEl) recEl.innerHTML = `<p><strong>Recommendation:</strong> ${signal.recommendation}</p>`;
        
        // Update guide based on strategy
        updateSignalGuide(signal);
    } catch (e) {
        console.error("Trading signal error:", e);
    }
}

async function loadTradingStrategies() {
    try {
        const res = await fetch(`${API}/api/trading-strategies`);
        const data = await res.json();
        const strategies = data.strategies;
        
        const selectedStrategy = localStorage.getItem("selectedTradingStrategy") || "conservative";
        const container = document.getElementById("strategy-buttons");
        
        if (container) {
            container.innerHTML = strategies.map(strat => `
                <button class="strategy-btn ${strat.id === selectedStrategy ? "active" : ""}" 
                        onclick="selectTradingStrategy(event, '${strat.id}')" 
                        title="${strat.best_for}">
                    <span class="strategy-btn-emoji">${strat.emoji}</span>
                    <span>${strat.name}</span>
                </button>
            `).join("");
        }

        // Cache strategies so the signal guide can read real thresholds
        strategiesCache = {};
        strategies.forEach(s => { strategiesCache[s.id] = s; });
        
        // Show info for selected strategy
        const selectedStrat = strategies.find(s => s.id === selectedStrategy);
        if (selectedStrat) {
            const infoCard = document.getElementById("strategy-info");
            if (infoCard) {
                infoCard.innerHTML = `
                    <p class="strategy-info-text">
                        <strong>${selectedStrat.emoji} ${selectedStrat.name}</strong><br/>
                        ${selectedStrat.description}<br/>
                        <em>Risk Level: ${selectedStrat.risk_level}</em>
                    </p>
                `;
            }
        }
    } catch (e) {
        console.error("Loading trading strategies error:", e);
    }
}

async function selectTradingStrategy(event, strategyId) {
    try {
        // Save to localStorage
        localStorage.setItem("selectedTradingStrategy", strategyId);
        
        // Update buttons
        document.querySelectorAll(".strategy-btn").forEach(btn => {
            btn.classList.remove("active");
        });
        event.target.closest(".strategy-btn").classList.add("active");
        
        // Reload signal, chart signals, and strategy info
        await Promise.all([
            loadTradingSignal(),
            loadTradingStrategies(),
            loadChart(),
        ]);
    } catch (e) {
        console.error("Strategy selection error:", e);
    }
}

function updateSignalGuide(signal) {
    const guideContainer = document.getElementById("signal-guide-container");
    if (!guideContainer) return;

    // Read real thresholds from the cached API response — no hardcoded values
    const strategy = signal.strategy;
    const cached   = strategiesCache[strategy];
    const buy  = cached ? cached.parameters.buy_threshold  : 20;
    const sell = cached ? cached.parameters.sell_threshold : 75;
    const hold = `${buy + 1}\u2013${sell - 1}`;

    guideContainer.innerHTML = `
        <div class="guide-item">
            <span class="emoji">🟢</span>
            <span class="label">BUY</span>
            <span class="description">Risk ≤ ${buy}. Markets are favorable for entry.</span>
        </div>
        <div class="guide-item">
            <span class="emoji">🟡</span>
            <span class="label">HOLD</span>
            <span class="description">Risk ${hold}. Wait for clearer signals.</span>
        </div>
        <div class="guide-item">
            <span class="emoji">🔴</span>
            <span class="label">SELL</span>
            <span class="description">Risk ≥ ${sell}. Consider reducing exposure.</span>
        </div>
    `;
}


// ============================================================
// PERFORMANCE METRICS
// ============================================================

async function loadPerformanceMetrics() {
    try {
        const res = await fetch(`${API}/api/performance`);
        const data = await res.json();
        const perf = data.performance;
        const dist = data.distribution;

        // Update cards
        if (document.getElementById("perf-predictions")) {
            document.getElementById("perf-predictions").textContent = perf.total_predictions || 0;
            document.getElementById("perf-if-avg").textContent   = perf.avg_if_score   != null ? Number(perf.avg_if_score).toFixed(1)   : "--";
            document.getElementById("perf-z-avg").textContent    = perf.avg_z_score    != null ? Number(perf.avg_z_score).toFixed(1)    : "--";
            document.getElementById("perf-lstm-avg").textContent = perf.avg_lstm_score != null ? Number(perf.avg_lstm_score).toFixed(1) : "--";
        }

        // Distribution table
        const distBody = document.getElementById("perf-distribution-body");
        if (distBody) {
            if (!dist || !Object.keys(dist).length) {
                distBody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--text-muted);padding:24px">No anomalies logged in the last 7 days.</td></tr>`;
            } else {
                distBody.innerHTML = Object.entries(dist).map(([level, d]) => `
                    <tr>
                        <td><strong>${level}</strong></td>
                        <td>${d.count}</td>
                        <td>${d.avg_score.toFixed(1)}</td>
                    </tr>
                `).join("");
            }
        }

        // Consensus
        const consensusFill = document.getElementById("consensus-fill");
        const consensusValue = document.getElementById("consensus-value");
        if (consensusFill && consensusValue) {
            const rate = perf.consensus_rate || 0;
            consensusFill.style.width = `${rate}%`;
            consensusValue.textContent = `${rate.toFixed(1)}%`;
        }
    } catch (e) {
        console.error("Performance metrics error:", e);
    }

    // Also load backtest in same call so the Performance tab is fully populated
    await loadBacktest();
}

// ============================================================
// SIGNAL ACCURACY BACKTEST
// ============================================================

async function loadBacktest() {
    try {
        const res  = await fetch(`${API}/api/backtest?days=30&coin=${currentCoin}`);
        const data = await res.json();

        const msgEl = document.getElementById("backtest-msg");

        if (!data.summary || Object.keys(data.summary).length === 0 || (data.total_signals || 0) === 0) {
            if (msgEl) {
                msgEl.textContent = "Not enough historical signal data yet. Come back after the system has been running for at least 24 hours.";
                msgEl.classList.remove("hidden");
            }
            return;
        }
        if (msgEl) msgEl.classList.add("hidden");

        const signals = ["BUY", "SELL", "HOLD"];
        signals.forEach(sig => {
            const key = sig.toLowerCase();
            const info = data.summary[sig] || {};
            const pct  = info.accuracy != null ? `${Number(info.accuracy).toFixed(1)}%` : "--";
            const correct = info.correct ?? "--";
            const total   = info.total   ?? "--";

            const pctEl    = document.getElementById(`bt-${key}-pct`);
            const countEl  = document.getElementById(`bt-${key}-counts`);
            if (pctEl)   pctEl.textContent   = pct;
            if (countEl) countEl.textContent = `${correct} / ${total} correct`;
        });
    } catch (e) {
        console.error("Backtest load error:", e);
    }
}

// ============================================================
// EXPORT FUNCTIONS
// ============================================================

function exportAnomalies() {
    window.location.href = `${API}/api/export/anomalies?limit=10000`;
}

function exportPriceData() {
    window.location.href = `${API}/api/export/price-data?days=90`;
}

async function clearAnomalyLogs() {
    if (!confirm("Permanently delete all anomaly logs for this coin? This cannot be undone.")) return;
    try {
        const res  = await fetch(`${API}/api/anomalies/clear?coin=${currentCoin}`, { method: "DELETE" });
        const data = await res.json();
        alert(`Cleared ${data.deleted} log entries.`);
        showAllSignals = false;
        const filterBtn = document.getElementById("signal-filter-btn");
        if (filterBtn) { filterBtn.textContent = "Show All Signals"; filterBtn.classList.remove("active"); }
        await loadAnomalies(1);
        await loadContinuitySignals(1);
    } catch (e) {
        alert("Failed to clear logs. Check console.");
        console.error(e);
    }
}

// ============================================================
// CHART
// ============================================================

async function loadChart() {
    try {
        const selectedStrategy = localStorage.getItem("selectedTradingStrategy") || "conservative";
        const res  = await fetch(`${API}/api/chart?range=${currentRange}&strategy=${selectedStrategy}&coin=${currentCoin}`);
        const data = await res.json();

        let forecast = [];
        if (showForecast) {
            try {
                const fRes  = await fetch(`${API}/api/forecast?steps=24&coin=${currentCoin}`);
                const fData = await fRes.json();
                forecast = fData.forecast || [];
            } catch (_) {}
        }

        renderChart(data.candles, data.anomalies || [], data.signals || [], forecast);
    } catch (e) {
        console.error("Chart load failed:", e);
    }
}

function setRange(range, btn) {
    currentRange = range;
    document.querySelectorAll(".range-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    loadChart();
}

function setChartType(type, btn) {
    currentChartType = type;
    document.querySelectorAll(".chart-type-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    loadChart();
}

function toggleOverlay(type) {
    if (type === "ma")        showMA        = !showMA;
    if (type === "bb")        showBB        = !showBB;
    if (type === "anomalies") showAnomalies = !showAnomalies;
    if (type === "signals")   showSignals   = !showSignals;
    if (type === "volume")    showVolume    = !showVolume;
    if (type === "forecast")  showForecast  = !showForecast;
    loadChart();
}

function renderChart(candles, anomalies = [], signals = [], forecast = []) {
    if (!candles || candles.length === 0) return;

    const timestamps = candles.map(c => c.timestamp);
    const opens      = candles.map(c => c.open);
    const highs      = candles.map(c => c.high);
    const lows       = candles.map(c => c.low);
    const closes     = candles.map(c => c.close);
    // DB stores USDT volume; convert to base-asset units by dividing by close price
    const volumes    = candles.map((c, i) => closes[i] > 0 ? c.volume / closes[i] : 0);

    // Candlestick trace
    const candlestick = {
        type:       "candlestick",
        x:          timestamps,
        open:       opens,
        high:       highs,
        low:        lows,
        close:      closes,
        name:       `${currentCoin}/USD`,
        increasing: { line: { color: "#22c55e" }, fillcolor: "#22c55e" },
        decreasing: { line: { color: "#ef4444" }, fillcolor: "#ef4444" },
    };

    // Build price trace — candlestick or line (close prices)
    let priceTrace;
    if (currentChartType === "line") {
        priceTrace = {
            type: "scatter", mode: "lines",
            x: timestamps, y: closes,
            name: `${currentCoin}/USD`,
            line: { color: "#3b82f6", width: 2 },
        };
    } else {
        priceTrace = candlestick;
    }

    const traces = [priceTrace];

    // Moving Average (20 period)
    if (showMA) {
        const ma = closes.map((_, i) => {
            if (i < 19) return null;
            const slice = closes.slice(i - 19, i + 1).filter(v => v != null && !isNaN(v));
            if (slice.length < 20) return null;
            return slice.reduce((a, b) => a + b, 0) / slice.length;
        });
        traces.push({
            type: "scatter", mode: "lines",
            x: timestamps, y: ma,
            name: "MA (20)",
            line: { color: "#3b82f6", width: 2 },
        });
    }

    // Bollinger Bands (20 period, 2 std dev) — expanding window for the first 19 candles
    if (showBB) {
        const bbMid = closes.map((_, i) => {
            const slice = closes.slice(Math.max(0, i - 19), i + 1).filter(v => v != null && !isNaN(v));
            if (slice.length < 2) return null;
            return slice.reduce((a, b) => a + b, 0) / slice.length;
        });
        const bbUpper = closes.map((_, i) => {
            const slice = closes.slice(Math.max(0, i - 19), i + 1).filter(v => v != null && !isNaN(v));
            if (slice.length < 2) return null;
            const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
            const std  = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length);
            return mean + 2 * std;
        });
        const bbLower = closes.map((_, i) => {
            const slice = closes.slice(Math.max(0, i - 19), i + 1).filter(v => v != null && !isNaN(v));
            if (slice.length < 2) return null;
            const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
            const std  = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length);
            return mean - 2 * std;
        });
        traces.push(
            { type: "scatter", mode: "lines", x: timestamps, y: bbUpper, name: "BB Upper", line: { color: "#eab308", width: 1.5, dash: "dot" } },
            { type: "scatter", mode: "lines", x: timestamps, y: bbLower, name: "BB Lower", line: { color: "#eab308", width: 1.5, dash: "dot" }, fill: "tonexty", fillcolor: "rgba(234,179,8,0.04)" },
            { type: "scatter", mode: "lines", x: timestamps, y: bbMid,   name: "BB Mid",   line: { color: "#eab308", width: 1.5 } }
        );
    }

    // Anomaly markers — drawn on price chart when the overlay toggle is active
    if (showAnomalies && anomalies.length > 0) {
        const levelColor = { High: "#ef4444", Critical: "#a855f7" };
        traces.push({
            type:          "scatter",
            mode:          "markers",
            x:             anomalies.map(a => a.timestamp),
            y:             anomalies.map(a => a.price),
            name:          "Anomalies",
            text:          anomalies.map(a => `${a.risk_level} Risk · ${a.score.toFixed(0)}/100`),
            hovertemplate: "%{text}<extra></extra>",
            marker: {
                symbol: "circle-open",
                size:   12,
                line:   { width: 2 },
                color:  anomalies.map(a => levelColor[a.risk_level] || "#ef4444"),
            },
        });
    }

    // Trading signal markers — BUY ▲, SELL ▼ only (HOLD is omitted from chart to reduce clutter)
    if (showSignals && signals.length > 0) {
        const sigConfig = {
            BUY:  { symbol: "triangle-up",   color: "#22c55e", yOffset:  1.004 },
            SELL: { symbol: "triangle-down",  color: "#ef4444", yOffset:  0.996 },
        };

        // Group by signal type so each gets its own legend entry
        ["BUY", "SELL"].forEach(type => {
            const group = signals.filter(s => s.signal === type);
            if (group.length === 0) return;
            const cfg = sigConfig[type];
            traces.push({
                type:          "scatter",
                mode:          "markers",
                x:             group.map(s => s.timestamp),
                y:             group.map(s => s.price * cfg.yOffset),
                name:          type,
                text:          group.map(s => `${type} · ${s.strategy} · risk ${s.risk_score.toFixed(0)} · conf ${s.confidence.toFixed(0)}%`),
                hovertemplate: "%{text}<extra></extra>",
                marker: {
                    symbol: cfg.symbol,
                    size:   10,
                    color:  cfg.color,
                    line:   { color: cfg.color, width: 1 },
                },
            });
        });
    }

    // LSTM forecast line — dotted purple line extending beyond the last candle
    if (showForecast && forecast.length > 0) {
        traces.push({
            type:          "scatter",
            mode:          "lines+markers",
            x:             forecast.map(f => f.timestamp),
            y:             forecast.map(f => f.price),
            name:          "LSTM Forecast",
            text:          forecast.map(f => `Forecast: $${Number(f.price).toLocaleString("en-US", { minimumFractionDigits: 2 })}`),
            hovertemplate: "%{text}<extra></extra>",
            line:  { color: "#a855f7", width: 2, dash: "dot" },
            marker: { size: 4, color: "#a855f7" },
        });
    }

    // Volume bars — rendered as semi-transparent bars on a secondary y-axis
    // capped at 95th-percentile so outlier seed-candle bars don't crush the scale
    let volMax = null;
    if (showVolume && volumes.some(v => v > 0)) {
        const sortedVols = [...volumes].filter(v => v > 0).sort((a, b) => a - b);
        const p95idx     = Math.floor(sortedVols.length * 0.95);
        volMax           = sortedVols[p95idx] * 1.5 || sortedVols[sortedVols.length - 1] * 1.5;

        traces.push({
            type:          "bar",
            x:             timestamps,
            y:             volumes,
            name:          "Volume",
            yaxis:         "y2",
            hovertemplate: `Vol: %{y:,.0f} ${currentCoin}<extra></extra>`,
            marker: {
                color: closes.map((c, i) =>
                    c >= opens[i] ? "rgba(34,197,94,0.8)" : "rgba(239,68,68,0.8)"
                ),
                line: {
                    width: 0.5,
                    color: closes.map((c, i) =>
                        c >= opens[i] ? "rgba(34,197,94,1)" : "rgba(239,68,68,1)"
                    ),
                },
            },
        });
    }

    // Determine the full x extent, extending to include forecast if shown
    const firstTs = timestamps[0];
    const lastTs  = showForecast && forecast.length > 0
        ? forecast[forecast.length - 1].timestamp
        : timestamps[timestamps.length - 1];

    // Day % change annotation (top-left corner, green/red)
    const annotations = [];
    if (lastChangePct !== null) {
        const pctColor = lastChangePct >= 0 ? "#22c55e" : "#ef4444";
        const pctSign  = lastChangePct >= 0 ? "+" : "";
        annotations.push({
            xref: "paper", yref: "paper",
            x: 0.01, y: 0.98,
            xanchor: "left", yanchor: "top",
            text: `<b>${pctSign}${lastChangePct.toFixed(2)}% (24h)</b>`,
            showarrow: false,
            font: { size: 13, color: pctColor },
            bgcolor: "rgba(6,10,15,0.65)",
            borderpad: 3,
        });
    }

    const layout = {
        paper_bgcolor: "#060a0f",
        plot_bgcolor:  "#060a0f",
        font:          { color: "#94a3b8", size: 11 },
        xaxis: {
            gridcolor:   "#1e2d3d",
            rangeslider: { visible: false },
            type:        "date",
            tickfont:    { size: 10 },
            range:       [firstTs, lastTs],
        },
        yaxis: {
            gridcolor:  "#1e2d3d",
            side:       "right",
            tickprefix: "$",
            tickfont:   { size: 10 },
            autorange:  true,
            domain:     [0.26, 1],   // leave bottom 26% for volume bars
        },
        // Secondary y-axis for volume — hidden scale, bottom 22% of chart
        yaxis2: {
            domain:         [0, 0.22],
            showgrid:       false,
            showticklabels: false,
            fixedrange:     true,
            range:          volMax ? [0, volMax] : undefined,
        },
        legend:    { bgcolor: "transparent", font: { size: 11 } },
        margin:    { l: 10, r: 70, t: 10, b: 30 },
        hovermode:   "x unified",
        autosize:    true,
        annotations: annotations,
    };

    Plotly.react("price-chart", traces, layout, {
        responsive: true,
        displayModeBar: "hover",
        modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
        displaylogo: false,
        scrollZoom: true,
    });
}

// ============================================================
// ANOMALIES TABLE
// ============================================================

async function loadAnomalies(page = 1) {
    currentPage = page;
    try {
        const qs   = `page=${page}&limit=20&coin=${currentCoin}${showAllSignals ? "&include_synthetic=true" : ""}`;
        const res  = await fetch(`${API}/api/anomalies?${qs}`);
        const data = await res.json();
        renderAnomalyTable(data.anomalies);
        renderPagination(data.page, data.total_pages);
        updateAnomalyBadge(data.total);

        // Update total anomalies count on model stats card
        const el = document.getElementById("stat-total-anomalies");
        if (el) el.textContent = data.total.toLocaleString();
    } catch (e) {
        console.error("Anomaly load failed:", e);
    }
}

function toggleSignalFilter(btn) {
    showAllSignals = !showAllSignals;
    btn.textContent = showAllSignals ? "Real Signals Only" : "Show All Signals";
    btn.classList.toggle("active", showAllSignals);
    currentPage = 1;
    loadAnomalies(1);
}

function renderAnomalyTable(anomalies) {
    const tbody = document.getElementById("anomaly-table-body");
    tbody.innerHTML = "";

    if (!anomalies || anomalies.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:#64748b;padding:40px">No risk signals detected yet.</td></tr>`;
        return;
    }

    anomalies.forEach(a => {
        const levelClass = `risk-${a.risk_level.toLowerCase()}`;
        const isExpanded = expandedAnomalyId === a.id;
        const isSynthetic = (a.signal_type === "synthetic");
        const synthLabel  = isSynthetic
            ? `<span style="font-size:10px;color:#64748b;background:rgba(100,116,139,0.15);border-radius:3px;padding:1px 5px;margin-left:5px">continuity</span>`
            : "";

        const row = document.createElement("tr");
        if (isSynthetic) row.style.opacity = "0.65";
        row.innerHTML = `
            <td><button class="expand-btn" onclick="toggleExplain(${a.id}, this)">${isExpanded ? "-" : "+"}</button></td>
            <td>${formatTimestamp(a.timestamp)}${synthLabel}</td>
            <td style="font-family:var(--font-mono)">$${Number(a.close_price).toLocaleString("en-US", { minimumFractionDigits: 2 })}</td>
            <td><strong>${a.risk_score.toFixed(1)}</strong></td>
            <td><span class="risk-badge ${levelClass}" style="font-size:11px;padding:3px 8px">${a.risk_level}</span></td>
            <td>${a.isolation_forest_score.toFixed(1)}</td>
            <td>${a.zscore_score.toFixed(1)} (${a.zscore_value}σ)</td>
            <td>${a.lstm_score.toFixed(1)}</td>
            <td>${a.confidence_level}</td>
            <td>${a.signal_strength || "--"}</td>
        `;
        tbody.appendChild(row);

        if (isExpanded) {
            const explainRow = document.createElement("tr");
            explainRow.className = "explain-row";
            explainRow.innerHTML = `<td colspan="10">${buildExplainHTML(a)}</td>`;
            tbody.appendChild(explainRow);
        }
    });
}

async function loadContinuitySignals(page = 1) {
    const tbody = document.getElementById("continuity-table-body");
    if (!tbody) return;
    try {
        const res = await fetch(`${API}/api/continuity-signals?page=${page}&limit=10&coin=${currentCoin}`);
        const data = await res.json();
        const rows = data.signals || [];
        if (!rows.length) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#64748b;padding:20px">No continuity signals recorded.</td></tr>`;
            return;
        }
        tbody.innerHTML = rows.map(r => `
            <tr>
                <td>${formatTimestamp(r.timestamp)}</td>
                <td>${Number(r.risk_score || 0).toFixed(1)}</td>
                <td>${r.risk_level || "--"}</td>
                <td>${r.signal_type || "synthetic"}</td>
                <td>${r.confidence_level || "Low"}</td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Continuity signal load failed:", e);
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#64748b;padding:20px">Failed to load continuity signals.</td></tr>`;
    }
}

function toggleExplain(id, btn) {
    expandedAnomalyId = expandedAnomalyId === id ? null : id;
    btn.textContent   = expandedAnomalyId === id ? "-" : "+";
    loadAnomalies(currentPage);
}

function buildExplainHTML(a) {
    return `
        <div class="explain-grid">
            <div class="explain-card">
                <h4>Score Breakdown</h4>
                <div class="score-bar-wrap">
                    <div class="score-bar-label"><span>Isolation Forest (25%)</span><span>${a.isolation_forest_score.toFixed(1)}</span></div>
                    <div class="score-bar-track"><div class="score-bar-fill bar-if" style="width:${a.isolation_forest_score}%"></div></div>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-label"><span>Z-Score (25%)</span><span>${a.zscore_score.toFixed(1)}</span></div>
                    <div class="score-bar-track"><div class="score-bar-fill bar-zs" style="width:${a.zscore_score}%"></div></div>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-label"><span>LSTM (50%)</span><span>${a.lstm_score.toFixed(1)}</span></div>
                    <div class="score-bar-track"><div class="score-bar-fill bar-lstm" style="width:${a.lstm_score}%"></div></div>
                </div>
            </div>
            <div class="explain-card">
                <h4>Isolation Forest</h4>
                <p>${a.if_reason || "--"}</p>
            </div>
            <div class="explain-card">
                <h4>Z-Score</h4>
                <p>${a.zscore_reason || "--"}</p>
            </div>
            <div class="explain-card">
                <h4>LSTM</h4>
                <p>${a.lstm_reason || "--"}</p>
                <p style="margin-top:6px;color:#64748b;font-size:11px">Predicted: $${Number(a.lstm_predicted_price).toLocaleString()}</p>
            </div>
        </div>
        <div class="summary-box">
            <strong>Summary:</strong> ${a.plain_english_summary}
            <p style="margin-top:6px;color:#64748b;font-size:11px">Type: ${a.signal_type || "real"} · Contributors: ${a.contributing_models || "None"}</p>
        </div>
    `;
}

function renderPagination(current, total) {
    const container = document.getElementById("pagination-controls");
    container.innerHTML = "";
    if (total <= 1) return;

    const prev = document.createElement("button");
    prev.className   = "page-btn";
    prev.textContent = "Prev";
    prev.disabled    = current === 1;
    prev.onclick     = () => loadAnomalies(current - 1);
    container.appendChild(prev);

    for (let i = 1; i <= total; i++) {
        if (total > 7 && Math.abs(i - current) > 2 && i !== 1 && i !== total) continue;
        const btn = document.createElement("button");
        btn.className   = `page-btn ${i === current ? "active" : ""}`;
        btn.textContent = i;
        btn.onclick     = () => loadAnomalies(i);
        container.appendChild(btn);
    }

    const next = document.createElement("button");
    next.className   = "page-btn";
    next.textContent = "Next";
    next.disabled    = current === total;
    next.onclick     = () => loadAnomalies(current + 1);
    container.appendChild(next);
}

function updateAnomalyBadge(total) {
    const badge = document.getElementById("anomaly-count-badge");
    if (total > 0) {
        badge.textContent = total;
        badge.classList.remove("hidden");
    } else {
        badge.classList.add("hidden");
    }
}

async function refreshAnomalyBadge() {
    try {
        const res  = await fetch(`${API}/api/anomalies?page=1&limit=1`);
        const data = await res.json();
        updateAnomalyBadge(data.total);
    } catch (e) {}
}

// ============================================================
// MODEL STATS
// ============================================================

async function loadModelStats() {
    try {
        const res  = await fetch(`${API}/api/models`);
        const data = await res.json();
        renderModelStats(data);
    } catch (e) {
        console.error("Model stats load failed:", e);
    }
}

function renderModelStats(data) {
    const registry = data.registry || [];
    const events   = data.events   || [];
    const currentAnomalyRate = data.current_anomaly_rate || 0;

    // Stat cards
    const lastRetrain = events.find(e => e.event_type === "retrain");
    const retainEl    = document.getElementById("stat-last-retrain");
    if (retainEl) retainEl.textContent = lastRetrain ? formatTimestamp(lastRetrain.created_at) : "Not yet";

    const latestModel  = registry[0];
    const sizeEl       = document.getElementById("stat-training-size");
    if (sizeEl && latestModel) sizeEl.textContent = `${latestModel.training_data_size.toLocaleString()} rows`;

    // Show CURRENT anomaly rate (more useful than training rate)
    const ifModel  = registry.find(r => r.model_type === "isolation_forest");
    const ifRateEl = document.getElementById("stat-if-rate");
    if (ifRateEl) ifRateEl.textContent = `${currentAnomalyRate.toFixed(1)}% (current)`;

    // Model registry table - Show note that training rate != current rate
    const regBody = document.getElementById("model-registry-body");
    if (regBody) {
        regBody.innerHTML = "";
        
        // Add a header row with explanation
        const headerNote = document.createElement("tr");
        headerNote.style.backgroundColor = "rgba(59,130,246,0.1)";
        headerNote.style.borderBottom = "2px solid var(--border)";
        headerNote.innerHTML = `
            <td colspan="5" style="font-size:11px; color:var(--text-muted); padding:8px">
                ℹ️ <strong>Anomaly Rate Note:</strong> Shows rate AT TRAINING TIME (% of training data that was anomalous). 
                See "Performance" tab for CURRENT anomaly detection rate. 0.0% = model learned baseline patterns well during training.
            </td>
        `;
        regBody.appendChild(headerNote);
        
        registry.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${r.model_type}</td>
                <td>${r.file_path.split(/[\\/]/).pop()}</td>
                <td>${formatTimestamp(r.trained_at)}</td>
                <td>${r.training_data_size.toLocaleString()}</td>
                <td title="Training-time anomaly rate, not current">${(r.anomaly_rate * 100).toFixed(1)}%</td>
            `;
            regBody.appendChild(tr);
        });
    }

    // Events table
    const eventsBody = document.getElementById("events-body");
    if (eventsBody) {
        eventsBody.innerHTML = "";
        events.slice(0, 30).forEach(e => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td style="white-space:nowrap">${formatTimestamp(e.created_at)}</td>
                <td><span class="event-type-${e.event_type}">${e.event_type}</span></td>
                <td style="color:#94a3b8">${e.message}</td>
            `;
            eventsBody.appendChild(tr);
        });
    }
}

// ============================================================
// ANOMALY BANNER
// ============================================================

function showAnomalyBanner(risk) {
    const banner = document.getElementById("anomaly-banner");
    const text   = document.getElementById("anomaly-banner-text");
    text.textContent = `Risk Alert: Score ${risk.risk_score.toFixed(0)}/100 (${risk.risk_level}) - ${risk.plain_english_summary}`;
    banner.classList.remove("hidden");
}

function dismissBanner() {
    document.getElementById("anomaly-banner").classList.add("hidden");
}

// ============================================================
// BROWSER NOTIFICATIONS
// ============================================================

function requestNotificationPermission() {
    if (!("Notification" in window)) return;
    if (Notification.permission === "granted") {
        notifPermission = true;
    } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then(p => {
            notifPermission = p === "granted";
        });
    }
}

function triggerNotification(risk) {
    if (!notifPermission) return;
    new Notification("BTC Risk Alert", {
        body: `Score: ${risk.risk_score.toFixed(0)}/100 (${risk.risk_level}). Possible anomaly detected.`,
    });
}

// ============================================================
// HELPERS
// ============================================================

function formatTimestamp(ts) {
    if (!ts) return "--";
    // Timestamps from the backend are UTC but lack a 'Z' suffix.
    // Appending 'Z' tells the browser to treat them as UTC so they
    // are converted to the user's local timezone for display.
    const utcTs = (ts.endsWith("Z") || ts.includes("+")) ? ts : ts + "Z";
    return new Date(utcTs).toLocaleString("en-US", {
        month:  "short",
        day:    "numeric",
        hour:   "2-digit",
        minute: "2-digit"
    });
}

// ============================================================
// BOOT
// ============================================================

// Resume polling the moment the tab becomes visible again.
// Without this, Chrome/Firefox leave the throttled timer pending
// and the dashboard can appear frozen for up to 60 seconds.
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
        // Run a poll immediately, then restart the normal schedule.
        stopPolling();
        runPoll().finally(() => startPolling());
    } else {
        // Tab is hidden — stop the timer so we don't waste requests.
        stopPolling();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    checkSeedingStatus();

    // FAQ accordion — click a question to expand/collapse the answer
    document.querySelectorAll(".faq-question").forEach(q => {
        q.addEventListener("click", () => {
            const item = q.closest(".faq-item");
            const isOpen = item.classList.contains("open");
            // Close all open items first
            document.querySelectorAll(".faq-item.open").forEach(i => i.classList.remove("open"));
            // Toggle clicked item
            if (!isOpen) item.classList.add("open");
        });
    });
});