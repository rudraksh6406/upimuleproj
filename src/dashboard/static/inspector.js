// MuleGuard Dashboard - Account Inspector Module
const emptyState = document.getElementById('inspector-empty');
const detailsState = document.getElementById('inspector-details');

const inspVpa = document.getElementById('insp-vpa');
const inspStatus = document.getElementById('insp-status');
const inspScorePct = document.getElementById('insp-score-pct');
const inspScoreBar = document.getElementById('insp-score-bar');
const inspExplanation = document.getElementById('insp-explanation');
const inspNeighbors = document.getElementById('insp-neighbors');

let currentInspectedVpa = null;

async function inspectAccount(vpa) {
    currentInspectedVpa = vpa;
    emptyState.style.display = 'none';
    detailsState.style.display = 'block';

    inspVpa.innerText = vpa;
    inspStatus.innerText = 'FETCHING...';

    try {
        const res = await fetch(`/api/accounts/${encodeURIComponent(vpa)}`);
        if (!res.ok) throw new Error('Not found');
        const data = await res.json();

        const score = data.risk_score || 0.0;
        const scorePct = Math.round(score * 100);
        inspScorePct.innerText = `${scorePct}%`;
        inspScoreBar.style.width = `${scorePct}%`;

        // Color coding score
        if (score >= 0.85) {
            inspScoreBar.style.backgroundColor = '#ef4444';
            inspStatus.innerText = 'FROZEN';
            inspStatus.style.color = '#ef4444';
            inspStatus.style.background = 'rgba(239, 68, 68, 0.2)';
        } else if (score >= 0.60) {
            inspScoreBar.style.backgroundColor = '#f59e0b';
            inspStatus.innerText = 'RESTRICTED';
            inspStatus.style.color = '#f59e0b';
            inspStatus.style.background = 'rgba(245, 158, 11, 0.2)';
        } else if (score >= 0.30) {
            inspScoreBar.style.backgroundColor = '#3b82f6';
            inspStatus.innerText = 'FLAGGED';
            inspStatus.style.color = '#3b82f6';
            inspStatus.style.background = 'rgba(59, 130, 246, 0.2)';
        } else {
            inspScoreBar.style.backgroundColor = '#10b981';
            inspStatus.innerText = 'NORMAL';
            inspStatus.style.color = '#10b981';
            inspStatus.style.background = 'rgba(16, 185, 129, 0.2)';
        }

        inspExplanation.innerText = data.explanation || 'Normal transaction flow.';

        // Render Recent Counterparts
        inspNeighbors.innerHTML = '';
        if (data.recent_transactions && data.recent_transactions.length > 0) {
            data.recent_transactions.forEach(t => {
                const row = document.createElement('div');
                row.className = 'nbr-row';
                row.innerHTML = `
                    <span>${t.direction === 'IN' ? '⬇️' : '⬆️'} ${t.counterpart}</span>
                    <span style="color: #94a3b8;">₹${t.amount.toLocaleString()}</span>
                `;
                inspNeighbors.appendChild(row);
            });
        } else {
            inspNeighbors.innerHTML = '<div style="font-size:11px;color:#64748b;">No recent transactions</div>';
        }

    } catch (e) {
        inspStatus.innerText = 'LOCAL NODE';
        inspScorePct.innerText = 'N/A';
        inspExplanation.innerText = 'Active transaction node in dynamic memory graph.';
    }
}

// Action buttons
document.getElementById('btn-action-freeze').addEventListener('click', () => sendAction('FREEZE'));
document.getElementById('btn-action-restrict').addEventListener('click', () => sendAction('RESTRICT'));
document.getElementById('btn-action-allow').addEventListener('click', () => sendAction('ALLOW'));

async function sendAction(action) {
    if (!currentInspectedVpa) return;
    try {
        await fetch('/api/accounts/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                account_vpa: currentInspectedVpa,
                action: action,
                reason: 'Operator intervention via MuleGuard Inspector'
            })
        });
        inspectAccount(currentInspectedVpa);
    } catch (e) {}
}

window.inspectAccount = inspectAccount;
