// MuleGuard Dashboard - Real-Time D3.js Force-Directed Graph Visualizer
const svg = d3.select("#network-svg");
const container = document.getElementById("graph-container");

let width = container.clientWidth || 800;
let height = container.clientHeight || 500;

svg.attr("viewBox", [0, 0, width, height]);

const g = svg.append("g");

// Zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on("zoom", (event) => {
        g.attr("transform", event.transform);
    });

svg.call(zoom);

document.getElementById("btn-reset-zoom").addEventListener("click", () => {
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
});

// Arrow markers for directed edges
svg.append("defs").selectAll("marker")
    .data(["arrow-normal", "arrow-mule"])
    .join("marker")
    .attr("id", d => d)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 20)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("fill", d => d === "arrow-mule" ? "#ef4444" : "#64748b")
    .attr("d", "M0,-5L10,0L0,5");

// Simulation setup
const simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(60))
    .force("charge", d3.forceManyBody().strength(-120))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(22));

let linkGroup = g.append("g").attr("class", "links");
let nodeGroup = g.append("g").attr("class", "nodes");

let graphData = { nodes: [], links: [] };

function getNodeColor(d) {
    if (d.action === "FREEZE" || d.risk_score >= 0.85) return "#ef4444";
    if (d.action === "RESTRICT" || d.risk_score >= 0.60) return "#f59e0b";
    if (d.action === "FLAG" || d.risk_score >= 0.30) return "#3b82f6";
    if (d.is_merchant) return "#06b6d4";
    return "#10b981";
}

function updateGraph(newData) {
    if (!newData || !newData.nodes) return;

    // Merge nodes preserving existing positions
    const nodeMap = new Map(graphData.nodes.map(d => [d.id, d]));
    
    graphData.nodes = newData.nodes.map(d => {
        const existing = nodeMap.get(d.id);
        return existing ? Object.assign(existing, d) : d;
    });

    graphData.links = newData.links.map(d => Object.assign({}, d));

    // Update Links
    const link = linkGroup.selectAll("line")
        .data(graphData.links, d => `${d.source.id || d.source}->${d.target.id || d.target}`)
        .join(
            enter => enter.append("line")
                .attr("stroke", d => d.amount > 20000 ? "#f43f5e" : "#475569")
                .attr("stroke-width", d => Math.min(4, Math.max(1, Math.log10(d.amount || 100))))
                .attr("stroke-opacity", 0.6)
                .attr("marker-end", "url(#arrow-normal)"),
            update => update,
            exit => exit.remove()
        );

    // Update Nodes
    const node = nodeGroup.selectAll("g")
        .data(graphData.nodes, d => d.id)
        .join(
            enter => {
                const nodeEnter = enter.append("g")
                    .call(drag(simulation))
                    .on("click", (event, d) => {
                        if (window.inspectAccount) window.inspectAccount(d.vpa || d.id);
                    });

                nodeEnter.append("circle")
                    .attr("r", d => d.is_merchant ? 14 : 10)
                    .attr("fill", d => getNodeColor(d))
                    .attr("stroke", "#ffffff")
                    .attr("stroke-width", 1.5)
                    .attr("stroke-opacity", 0.6);

                nodeEnter.append("text")
                    .text(d => {
                        const parts = (d.vpa || d.id).split("@");
                        return parts[0].length > 10 ? parts[0].substring(0, 9) + "…" : parts[0];
                    })
                    .attr("x", 14)
                    .attr("y", 4)
                    .attr("fill", "#94a3b8")
                    .attr("font-size", "10px")
                    .attr("font-family", "monospace");

                return nodeEnter;
            },
            update => {
                update.select("circle")
                    .transition().duration(300)
                    .attr("fill", d => getNodeColor(d));
                return update;
            },
            exit => exit.remove()
        );

    simulation.nodes(graphData.nodes).on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    simulation.force("link").links(graphData.links);
    simulation.alpha(0.3).restart();
}

function drag(simulation) {
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }
    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

// WebSocket Connection for Live Feed
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === "GRAPH_UPDATE") {
                updateGraph(msg.data);
            } else if (msg.type === "ALERT" && window.addAlertItem) {
                window.addAlertItem(msg.data);
            }
        } catch (e) {}
    };

    socket.onclose = () => {
        setTimeout(initWebSocket, 3000);
    };
}

// Periodic Polling Fallback
async function fetchGraphData() {
    try {
        const res = await fetch("/api/graph?max_nodes=60");
        if (!res.ok) return;
        const data = await res.json();
        updateGraph(data);
    } catch (e) {}
}

window.addEventListener("resize", () => {
    width = container.clientWidth;
    height = container.clientHeight;
    svg.attr("viewBox", [0, 0, width, height]);
    simulation.force("center", d3.forceCenter(width / 2, height / 2));
    simulation.alpha(0.3).restart();
});

fetchGraphData();
setInterval(fetchGraphData, 3000);
initWebSocket();
