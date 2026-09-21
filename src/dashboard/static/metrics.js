// MuleGuard Dashboard - Metrics & Polling Module
let metricsData = {
    totalTxns: 0,
    flaggedMules: 0,
    frozenMules: 0,
    latencyMs: 11.4
};

async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) return;
        const data = await response.json();
        
        document.getElementById('metric-total-txns').innerText = data.total_transactions.toLocaleString();
        document.getElementById('metric-flagged-mules').innerText = data.total_flagged.toLocaleString();
        document.getElementById('metric-frozen-mules').innerText = data.total_frozen.toLocaleString();
        document.getElementById('metric-latency').innerText = data.avg_latency_ms;
    } catch (e) {
        // Fallback for offline / simulation
    }
}

// Poll metrics every 2 seconds
setInterval(fetchMetrics, 2000);
fetchMetrics();
