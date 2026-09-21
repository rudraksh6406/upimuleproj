// MuleGuard Dashboard - Alerts Feed Module
const alertsListEl = document.getElementById('alerts-list');
const alertBadgeEl = document.getElementById('alert-badge');
let alertCount = 0;

function addAlertItem(alert) {
    alertCount++;
    alertBadgeEl.innerText = `${alertCount} Live`;

    const item = document.createElement('div');
    const isCritical = alert.action === 'FREEZE' || alert.alert_level === 'critical';
    item.className = `alert-item ${isCritical ? 'critical' : ''}`;
    
    const tagClass = alert.action === 'FREEZE' ? 'tag-freeze' : (alert.action === 'RESTRICT' ? 'tag-restrict' : 'tag-flag');
    
    item.innerHTML = `
        <div class="alert-top">
            <span class="alert-vpa">${alert.account_vpa || alert.account_id}</span>
            <span class="alert-action-tag ${tagClass}">${alert.action}</span>
        </div>
        <div class="alert-text">${alert.explanation || 'Suspicious temporal graph anomaly detected.'}</div>
    `;

    item.addEventListener('click', () => {
        if (window.inspectAccount) {
            window.inspectAccount(alert.account_vpa || alert.account_id);
        }
    });

    alertsListEl.insertBefore(item, alertsListEl.firstChild);
    
    // Keep max 30 items
    if (alertsListEl.children.length > 30) {
        alertsListEl.removeChild(alertsListEl.lastChild);
    }
}

async function fetchInitialAlerts() {
    try {
        const res = await fetch('/api/alerts?limit=15');
        if (!res.ok) return;
        const alerts = await res.json();
        alerts.forEach(addAlertItem);
    } catch (e) {}
}

fetchInitialAlerts();
window.addAlertItem = addAlertItem;
