# Product Requirements Document (PRD)
# MuleGuard — Real-Time UPI Mule Account Detection using Temporal Graph Neural Networks

**Document Version:** 2.0
**Date:** 21 September 2026
**Project Type:** Personal / Final-Year Major Project (B.Tech / B.E. Computer Science)
**Project Duration:** 14–16 weeks (one semester)
**Team Size:** 1 (solo, designed for individual execution)
**Constraint:** Fully self-contained — no proprietary datasets, no internal dashboards, no Razorpay-internal APIs. Everything buildable from scratch using only open-source tools, public data standards, and synthetic data.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Objectives & Success Criteria](#3-objectives--success-criteria)
4. [Literature Review & Background](#4-literature-review--background)
5. [System Architecture](#5-system-architecture)
6. [Data Layer — Synthetic UPI Transaction Generator](#6-data-layer--synthetic-upi-transaction-generator)
7. [Temporal Graph Neural Network — Model Design](#7-temporal-graph-neural-network--model-design)
8. [Real-Time Inference Pipeline](#8-real-time-inference-pipeline)
9. [Guardian — Decision & Action Engine](#9-guardian--decision--action-engine)
10. [Dashboard & Visualization](#10-dashboard--visualization)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Tech Stack](#12-tech-stack)
13. [Project Structure & Repository Layout](#13-project-structure--repository-layout)
14. [Project Milestones & Timeline](#14-project-milestones--timeline)
15. [Deliverables](#15-deliverables)
16. [Academic Requirements Mapping](#16-academic-requirements-mapping)
17. [Risks & Mitigations](#17-risks--mitigations)
18. [Future Work](#18-future-work)
19. [References](#19-references)
20. [Appendix A: Glossary](#appendix-a-glossary)
21. [Appendix B: Configuration Files](#appendix-b-configuration-files)
22. [Appendix C: Real-World Fraud Patterns Observed in Industry](#appendix-c-real-world-fraud-patterns-observed-in-industry)
23. [Appendix D: Sample Transaction Event Schema](#appendix-d-sample-transaction-event-schema)
24. [Appendix E: Mathematical Formulation of TGN](#appendix-e-mathematical-formulation-of-tgn)

---

## 1. Executive Summary

### 1.1 What is MuleGuard?

MuleGuard is a real-time fraud detection system that identifies mule accounts in UPI (Unified Payments Interface) transaction networks using Temporal Graph Neural Networks (TGNNs). It processes live transaction streams, constructs a dynamic graph of money flows, and uses a trained TGNN model to score every account's probability of being a mule — in real time, as transactions happen. The system is entirely self-contained: it generates its own synthetic UPI transaction data with planted mule networks, trains the model from scratch, runs a real-time inference pipeline, hosts a live dashboard with graph visualization, and evaluates itself against multiple baselines — all using only open-source tools and a single GPU (or free Google Colab).

### 1.2 Why does this matter?

UPI processes over 500 million transactions daily in India. Mule accounts — accounts used to receive and forward illicit funds, acting as conduits in money laundering chains — are a growing problem flagged by the RBI, I4C (Indian Cyber Crime Coordination Centre), and financial institutions nationwide. The RBI has specifically mandated that all regulated entities shift from reactive fraud management to proactive risk mitigation, with dedicated working groups on money mule account detection. Current detection systems rely on rule-based heuristics (velocity checks, threshold alerts, hotspot pincode monitoring) that are reactive, easily evaded, and produce high false-positive rates. A TGNN-based approach learns the structural and temporal patterns of mule behavior directly from the transaction graph, enabling detection of novel fraud patterns that no human thought to write a rule for.

### 1.3 What makes this project full-marks worthy?

- **Technical depth:** Combines graph theory, deep learning (GNNs), temporal modeling, real-time streaming, and full-stack development — spanning ML, systems, and visualization.
- **Novelty:** TGNNs applied to UPI mule detection is a genuinely novel application. Most published work uses static GNNs or traditional ML on tabular features. The temporal dimension — tracking how the money-flow graph evolves over time — is the key innovation.
- **Real-world relevance:** Solves a problem with massive scale and societal impact. India's financial fraud losses are measured in thousands of crores annually. The RBI's Harbinger 2024 innovation challenge specifically included mule account detection as a problem statement.
- **Self-contained:** No dependency on any proprietary API, internal dataset, or company infrastructure. Everything runs on a laptop or free cloud resources. This is critical for a personal/academic project.
- **Demonstrability:** The live dashboard with real-time graph visualization and mule detection is visually compelling and easy to understand for examiners.
- **Completeness:** End-to-end system — data generation, model training, real-time inference, decision engine, dashboard, evaluation — not a partial prototype.

### 1.4 Key Design Principle: Zero External Dependencies

This project must be buildable by anyone, anywhere, with no access to payment gateway data, bank APIs, or proprietary fraud labels. Everything is synthetic, open-source, and self-hosted:

| What real systems use | What this project uses instead |
|---|---|
| Real UPI transaction data from NPCI/banks | Synthetic data generator producing realistic UPI transactions with planted mule patterns |
| Razorpay Shield / Sardine / Stripe Radar fraud scores | Custom TGNN model trained from scratch on synthetic labels |
| Internal risk ops dashboards (admin-dashboard) | Custom FastAPI + React dashboard, built from scratch |
 CPFIR / I4C / MNRL consortium data | Simulated mule account registry within synthetic data |
| Device fingerprinting from third-party vendors (ThreatMetrix, etc.) | Simulated device fingerprints within synthetic data |
| Kafka cluster on production infra | Python `asyncio.Queue` or local Kafka (docker-compose) |
| Redis cluster for real-time state | Local Redis instance (docker-compose) or in-memory dict |

---

## 2. Problem Statement

### 2.1 The Mule Account Problem

A mule account is a bank account or UPI-linked account used by fraudsters to receive stolen money and forward it onward, often through multiple hops, to obscure the money's origin. Mule accounts are typically:

- **Compromised accounts** belonging to legitimate users whose credentials have been phished. The user may not even know their account is being used as a conduit.
- **Synthetic accounts** created with stolen KYC documents. Fraudsters use forged identity documents to open accounts in the names of victims who may not even exist or who exist but are unaware.
- **Witting mules** — people paid a commission to let their account be used for forwarding illicit funds. These are often recruited through fake job postings, gaming schemes, or social engineering.
- **Gaming/betting mules** — accounts registered as merchants or individuals processing proceeds from online betting scams, often registered under the same residential address.

The transaction patterns that reveal mule behavior are **graph-structural** and **temporal**:

| Pattern | Description | Real-World Inspiration |
|---|---|---|
| Fan-out | One source sends money to many accounts simultaneously within minutes | Scammer disperses stolen funds across 20+ mule accounts to distribute risk |
| Fan-in | Many sources send money to a single aggregator account | Mule accounts all forward to one exit account controlled by the fraudster |
| Layering chain | A → B → C → D — each hop obscures origin, each within minutes | Classic money laundering layering; 3–5 hops makes tracing extremely difficult |
| Burst activation | Dormant account suddenly receives and forwards within minutes | Account inactive for 60+ days, then 15 transactions in 30 minutes, then dormant again |
| Coordinated activation | Multiple mule accounts activate simultaneously from the same source | 10–20 accounts all receive from same source within same hour, all forward to same destination |
| Cyclic flow | Money returns to origin through a cycle of accounts | Wash-trade pattern; money laundering where funds cycle to appear legitimate |
| Velocity anomaly | Account receives and forwards with near-zero holding time | Funds in → funds out within 2–10 minutes; no legitimate user behaves this way |
| Device sharing | Multiple accounts accessed from the same device fingerprint | Fraudster controlling 50+ mule accounts from a single device |
| Amount clustering | Transactions at near-identical amounts across multiple accounts | Bot-driven card testing where 99.8% of transactions are exactly ₹211 |
| Gaming/betting funnel | Multiple accounts at same address processing gaming transactions | Real case: mule accounts registered under same residential address for betting proceeds |

### 2.2 Why Rule-Based Systems Fall Short

Rule-based systems detect known patterns using fixed thresholds (e.g., "flag accounts with >10 incoming transactions in 5 minutes"). They fail because:

- Fraudsters learn the thresholds and stay just below them. If the rule fires at 10 transactions, the fraudster sends 9.
- New mule patterns have no corresponding rule. A novel layering topology (e.g., a diamond pattern instead of a chain) is invisible to velocity rules.
- High false-positive rates — legitimate merchants receiving many payments during a sale look like fan-in. A user paying multiple vendors looks like fan-out.
- No understanding of graph context — an account is evaluated in isolation, not as part of a money-flow network. The rule sees "10 incoming transactions" but not "these 10 senders all received from the same source 5 minutes ago."
- Rules are whack-a-mole. Each new fraud pattern requires a human to identify it, write a rule, test it, and deploy it — a cycle that takes hours to days. By then, the fraudster has moved on.

Real-world observation: A device sending 9,500 requests with 7,500 cards, 8,500 email addresses, and 7,100 IP addresses across 105 countries — but at most ~50 requests per merchant to spread the load — is invisible to per-merchant rules. Only a cross-merchant graph view reveals the coordinated attack.

### 2.3 Why TGNNs

TGNNs address both dimensions that rules miss:

- **Graph structure:** The model sees the full neighborhood of each account — who sends to it, who it sends to, and the broader topology. A fan-out pattern is visible in the graph structure even if individual transaction counts stay below thresholds. The model learns that "account X has 15 neighbors, 12 of which were dormant before receiving from X" is suspicious — a rule would need to be explicitly written for this.
- **Temporal evolution:** The model tracks how the graph changes over time. A dormant account that suddenly bursts with activity is a strong mule signal — but only if you're modeling the temporal dimension, not just a static snapshot. The TGN architecture maintains a persistent memory per node that captures the account's entire interaction history, enabling it to distinguish "dormant then burst" from "consistently active."

### 2.4 Why Not Just Use Static GNNs?

Static GNNs (GCN, GraphSAGE, GAT) operate on a single snapshot of the graph. They can see structure but not evolution. Consider: an account that has been dormant for 90 days and then receives 10 transactions in 5 minutes. In a static snapshot, this looks identical to an active account that receives 10 transactions in 5 minutes as part of its normal behavior. Only the temporal history — the 90 days of dormancy followed by sudden activation — distinguishes the mule from the legitimate user. TGNNs capture this; static GNNs cannot.

---

## 3. Objectives & Success Criteria

### 3.1 Primary Objectives

1. **Design and implement a synthetic UPI transaction data generator** that produces realistic transaction streams with planted mule networks and known ground-truth labels. This generator must produce data that mimics real UPI patterns (VPA formats, transaction amounts, time-of-day distributions, P2P vs P2M ratios) closely enough that models trained on synthetic data would transfer to real data.

2. **Build and train a Temporal Graph Neural Network (TGN)** that classifies accounts as mule or legitimate in real time, using both graph-structural and temporal features. The model must process transactions in temporal order and maintain per-node memory states.

3. **Implement a real-time inference pipeline** that processes incoming transactions, updates the graph and node memories, and produces mule probability scores with sub-second latency.

4. **Build Guardian**, a decision engine that translates model scores into actions (flag, freeze, allow) with configurable thresholds, human-readable explanations, and a human-in-the-loop review path.

5. **Create a live monitoring dashboard** that visualizes the transaction graph in real time, shows risk scores per account, displays flagged mule subgraphs, and presents evaluation metrics.

6. **Evaluate the system rigorously** against six baselines (rule-based, logistic regression, XGBoost, static GCN, static GraphSAGE, TGAT) on precision, recall, F1, AUC-ROC, AUC-PR, false-positive rate, detection latency, and throughput. Include ablation studies and per-pattern breakdowns.

### 3.2 Success Criteria

| Criterion | Target | Measurement |
|---|---|---|
| Mule detection recall | ≥ 90% on held-out test set | Offline evaluation on test split |
| Precision | ≥ 85% on held-out test set | Offline evaluation on test split |
| False-positive rate | ≤ 5% | Offline evaluation on test split |
| Detection latency (transaction arrival → flag) | < 500ms (p95) | Timed during live demo |
| TGNN outperforms rule-based baseline | ≥ 15 percentage points higher F1 | Paired comparison on test set |
| TGNN outperforms static GNN baseline | ≥ 8 percentage points higher F1 | Paired comparison on test set |
| Dashboard renders live graph at | ≥ 30 FPS with 1,000+ nodes | Browser DevTools FPS counter |
| System handles throughput | ≥ 100 transactions/second | Timed during pipeline stress test |
| Code quality | Modular, documented, unit-tested, version-controlled | pytest coverage ≥ 80%, linted |
| Report quality | IEEE-format, 20+ pages, with literature review | Academic standard |
| Demo | Live, end-to-end, visually compelling | Rehearsed, runs on clean machine |
| Reproducibility | One-command setup via Docker | `docker-compose up` starts everything |

---

## 4. Literature Review & Background

### 4.1 Graph Neural Networks for Fraud Detection

**GCN (Graph Convolutional Network, Kipf & Welling 2017)** — spectral-based convolution on graphs. Applied to fraud detection by treating the transaction graph as static and classifying nodes based on their neighborhood aggregation. Limitation: ignores temporal dynamics — the graph is treated as a fixed snapshot. A fraudster who changes behavior over time is indistinguishable from their historical self.

**GraphSAGE (Hamilton et al. 2017)** — inductive representation learning via neighborhood sampling. Scales to large graphs by sampling a fixed-size neighborhood rather than aggregating over all neighbors. Still static — no temporal awareness. Widely used as a baseline in fraud detection research.

**GAT (Graph Attention Network, Veličković et al. 2018)** — attention-based message passing. Weights neighbors by learned importance rather than treating all neighbors equally. Applied to fraud by attending to suspicious neighbors more heavily. The attention mechanism provides a degree of explainability — you can see which neighbors contributed to a prediction. Still static.

**GNN-based fraud detection in practice:** Industry deployments (Block, Sardine, Razorpay's Shield) use graph-based features and GNN embeddings as inputs to fraud scoring models. However, most published work treats the graph as static or uses fixed temporal windows. The temporal graph network approach — where the model processes interactions as a continuous stream and maintains memory — is not yet standard in production fraud systems.

### 4.2 Temporal Graph Neural Networks

**TGN (Temporal Graph Networks, Rossi et al. 2020)** — the foundational TGNN architecture. Maintains a memory module per node that is updated with each interaction. Uses a message function, a memory updater (GRU/RNN), and a temporal attention module for embedding generation at any point in time. Designed for continuous-time dynamic graphs — exactly the UPI transaction setting where transactions arrive as a stream and the graph evolves continuously.

Key innovation: the per-node memory acts as a compressed summary of the node's entire interaction history. When a dormant account suddenly activates, its memory state — encoding 90 days of dormancy — is available to the model. This is impossible with snapshot-based approaches.

**TGAT (Temporal Graph Attention, Xu et al. 2020)** — extends GAT with temporal encoding. Uses functional time encoding based on Bochner's theorem to capture relative temporal positions. Produces time-aware node embeddings. Simpler than TGN (no persistent memory) but captures some temporal information through the time encoding.

**JODIE (Kumar et al. 2019)** — designed for user-item interaction prediction (e.g., Reddit users and subreddits). Uses a recurrent update mechanism where user and item embeddings are updated based on each interaction. Applicable to user-user UPI transactions with adaptation. JODIE's concept of "projecting" a user's embedding into the future to predict their next interaction is relevant for mule detection — a mule's projected trajectory differs from a legitimate user's.

**TGN vs. TGAT:** TGN maintains a persistent memory per node, making it more expressive for long-range temporal patterns (e.g., burst-dormant behavior spanning months). TGAT is simpler but requires windowed snapshots and loses information between windows. For this project, TGN is the primary architecture; TGAT serves as a baseline for comparison — the ablation "TGN without memory" is effectively TGAT, and the performance gap quantifies the value of persistent memory.

### 4.3 Mule Account Detection — Existing Approaches

- **Traditional ML (Logistic Regression, Random Forest, XGBoost):** Uses tabular per-account features (transaction count, average amount, velocity, account age, dormancy period). Cannot capture graph structure — an account with 10 incoming transactions looks the same whether those 10 senders are independent or all received from the same source. This is Baseline 1 and 2.

- **Rule-based:** Velocity thresholds (e.g., >10 txns in 5 minutes), fan-out/fan-in limits, dormancy-then-activity rules, hotspot pincode checks, device sharing thresholds. This is what most production systems use today. This is Baseline 3.

- **Static GNNs (GCN, GraphSAGE):** Capture structure but not temporal evolution. A snapshot of the graph at time T shows the fan-out pattern but not the 90 days of dormancy that preceded it. These are Baselines 4 and 5.

- **TGNNs (this project):** Captures both structure and temporal evolution. Novel contribution: applying TGN to UPI mule detection with real-time inference, synthetic data generation, and a complete decision engine.

### 4.4 UPI-Specific Context

- UPI uses VPA (Virtual Payment Address) identifiers — `name@bank` format (e.g., `john@okhdfcbank`, `merchant@axisb`)
- Transactions are real-time, 24x7, IMPS-rail, typically completing in under 5 minutes or expiring
- Typical transaction values range from ₹1 to ₹2,00,000 (UPI per-transaction limit)
- NPCI (National Payments Corporation of India) operates the UPI switch
- Major UPI apps: PhonePe, Google Pay, Paytm, CRED, Amazon Pay, BHIM, WhatsApp Pay
- P2P (person-to-person) and P2M (person-to-merchant) are the two primary transaction types
- No public dataset of UPI mule accounts exists — synthetic data generation is mandatory
- RBI's MNRL (Mobile Number Revocation List) database, integrated by payment entities, screens against mobile numbers revoked by TRAI for fraud/cybercrime — demonstrating the regulatory emphasis on mule account detection
- RBI has specifically flagged "hotspot areas" where mule account onboarding is concentrated, requiring enhanced due diligence for accounts from specific pin codes

### 4.5 Gap This Project Fills

No published work applies TGNNs specifically to UPI mule account detection with real-time inference. Existing TGNN fraud work focuses on credit card fraud (tabular + graph) or cryptocurrency money laundering (Bitcoin transaction graphs). UPI's unique characteristics — real-time P2P, VPA-based addressing, high velocity, Indian banking context, the specific mule patterns observed in the Indian fintech ecosystem — make this a novel application domain.

Additionally, this project addresses the practical gap of building the entire pipeline — from data generation to real-time inference to visualization — as a self-contained system, demonstrating end-to-end competence rather than just model training.

### 4.6 Industry Patterns That Informed This PRD

Research for this PRD included analysis of real-world fraud detection discussions from fintech engineering teams. Key patterns observed:

1. **Device fingerprinting is the strongest single signal** — in production, device fingerprint contributes roughly 44% of a fraud model's discriminative signal. Multiple accounts on the same device is the single biggest red flag for mule operations.

2. **Cross-merchant visibility is critical** — a fraudster testing 200+ cards across 18 merchants is invisible to any single merchant's per-transaction view. Only a graph that spans all merchants reveals the coordinated pattern.

3. **Card-testing attacks have distinct signatures** — 527 transactions from 527 distinct IPs, 416 distinct user agents, all routing through the same Cardinal/Centinel platform, all at near-identical amounts. The graph view reveals that all these "diverse" transactions originate from a single device.

4. **Anomaly in transaction pattern is the most common risk flag** — "anomaly_in_transaction_pattern" is the primary reason code used by risk teams to suspend merchants. This validates the TGNN approach: the model learns what "normal" looks like and flags anomalies structurally.

5. **Coordinated mule rings share infrastructure** — same residential address, same device, same bank account for settlements. The graph captures these shared-attribute edges naturally.

6. **Burst patterns are the strongest temporal signal** — dormant accounts that suddenly receive and forward within minutes are almost always mules. The TGN memory module is specifically designed to capture this pattern.

7. **Refund fraud and chargeback fraud have temporal signatures** — legitimate refunds cluster around 3–10 days after purchase; fraud refunds come within hours, before the card dispute window closes.

8. **RBI's regulatory framework increasingly mandates proactive detection** — the shift from reactive to proactive is explicitly required, and graph-based approaches are the natural next step beyond rules.

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MuleGuard System                                 │
│                                                                          │
│  ┌──────────────┐                                                       │
│  │  Synthetic    │    Transactions (JSON events)                        │
│  │  Data         │──────────────────────────────────────────┐           │
│  │  Generator    │                                          │           │
│  └──────────────┘                                          ▼           │
│                                               ┌────────────────────┐    │
│                                               │  Stream Source     │    │
│                                               │  (asyncio.Queue   │    │
│                                               │   or local Kafka)  │    │
│                                               └────────┬───────────┘    │
│                                                         │                │
│                                                         ▼                │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Temporal Graph Engine                             │       │
│  │                                                                │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐    │       │
│  │  │  Dynamic     │  │  TGN        │  │  Mule Probability  │    │       │
│  │  │  Graph       │  │  Inference  │──▶│  Score             │    │       │
│  │  │  Builder     │  │  Engine     │  │  (per account)     │    │       │
│  │  │  (in-memory) │  │             │  └────────┬───────────┘    │       │
│  │  └──────┬───────┘  └─────────────┘           │                │       │
│  │         │                                    │                │       │
│  │         ▼                                    ▼                │       │
│  │  ┌─────────────┐                    ┌────────────────────┐    │       │
│  │  │  Node        │                    │  Guardian           │    │       │
│  │  │  Memory      │                    │  Decision Engine    │    │       │
│  │  │  Store       │                    │  (thresholds,       │    │       │
│  │  │  (GRU states)│                    │   actions,          │    │       │
│  │  └─────────────┘                    │   explanations)     │    │       │
│  │                                      └────────┬───────────┘    │       │
│  └───────────────────────────────────────────────┼────────────────┘       │
│                                                   │                       │
│                              ┌────────────────────┼───────────────────┐   │
│                              ▼                    ▼                   │   │
│                    ┌──────────────────┐  ┌──────────────────────┐    │   │
│                    │  Live Dashboard  │  │  Evaluation Engine    │    │   │
│                    │  (FastAPI +      │  │  (metrics, baselines, │    │   │
│                    │   React + D3.js) │  │   ablations, plots)   │    │   │
│                    │                  │  │                        │    │   │
│                    │  - Graph Viz     │  │  - Precision/Recall    │    │   │
│                    │  - Alert Feed    │  │  - F1, AUC-ROC, AUC-PR │    │   │
│                    │  - Metrics Panel │  │  - Per-pattern recall  │    │   │
│                    │  - Account Info  │  │  - Latency histogram   │    │   │
│                    └──────────────────┘  └──────────────────────┘    │   │
│                                                                    │   │
│                    ┌──────────────────────────────────────────────┐ │   │
│                    │  Persistence Layer                            │ │   │
│                    │  - SQLite (audit logs, decisions, metrics)   │ │   │
│                    │  - Redis (real-time graph state, node memory)│ │   │
│                    │  - Filesystem (model checkpoints, plots)     │ │   │
│                    └──────────────────────────────────────────────┘ │   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow — End to End

1. **Data Generator** produces a stream of transaction events (JSON) with fields: `txn_id, sender_vpa, receiver_vpa, amount, timestamp, device_id, merchant_category, txn_type, sender_kyc_type, receiver_kyc_type, is_mule_sender, is_mule_receiver, mule_pattern_type`

2. **Stream Source** (asyncio.Queue or local Kafka) buffers transactions and feeds them to the pipeline in temporal order.

3. **Temporal Graph Builder** consumes each transaction, adds the edge to the in-memory dynamic graph, updates node features (degree, velocity, amount sums), and manages the sliding window (evicts nodes/edges older than 24 hours).

4. **TGN Inference Engine** processes each batch of transactions:
   - Updates node memories (GRU) for all nodes involved in the batch
   - Computes embeddings (temporal graph attention) for affected nodes
   - Runs the classifier MLP to produce mule probability scores

5. **Guardian Decision Engine** receives scores, applies threshold logic (with account-age adjustments), and produces actions: ALLOW, FLAG, RESTRICT, or FREEZE. Generates human-readable explanations from attention weights.

6. **Dashboard** receives updates via WebSocket: new graph edges (for visualization), alerts (for the alert feed), and metrics (for the metrics panel).

7. **Evaluation Engine** runs in parallel during the demo, computing rolling metrics (precision, recall, F1, FPR) using the ground-truth labels from the data generator.

8. **Persistence Layer** logs all decisions to SQLite, caches graph state in Redis, and saves model checkpoints and evaluation plots to the filesystem.

### 5.3 Component Breakdown

#### Component 1: Synthetic Data Generator
**Responsibility:** Generate a realistic stream of UPI transactions with a known mix of legitimate and mule accounts. Produces ground-truth labels for training and evaluation. This is a core deliverable — the quality of the entire system depends on the realism of the synthetic data.

**Key design decisions:**
- VPA format must match real UPI conventions (`name@okhdfcbank`, `name@okaxis`, `name@ybl`, etc.)
- Transaction amount distribution must match real UPI (log-normal, median ~₹100, heavy tail to ₹2L)
- Time-of-day distribution must be bimodal (lunch + evening peaks)
- Mule patterns must be subtle enough to be challenging but distinct enough to be learnable
- 5% mule ratio matches realistic fraud prevalence (not too imbalanced, not too easy)

#### Component 2: Temporal Graph Builder
**Responsibility:** Maintain an in-memory dynamic graph that grows as transactions arrive. Each node is an account (VPA) with features; each edge is a transaction with a timestamp and attributes.

**Key design decisions:**
- Graph stored as adjacency list in Python dict (or DGL/PyG `HeteroData` for training)
- Sliding window of last 24 hours for active subgraph (keeps memory bounded)
- Node features updated incrementally on each transaction (no full recomputation)
- Node memory (TGN): each node maintains a GRU-updated hidden state capturing its interaction history

#### Component 3: TGN Model
**Responsibility:** Given the current graph state + node memories, produce a mule probability score for any account.

**Architecture:** TGN with 5 modules (detailed in Section 7).

#### Component 4: Guardian Decision Engine
**Responsibility:** Translate mule probability scores into actions. Include configurable thresholds, account-age adjustments, and explanation generation.

#### Component 5: Dashboard
**Responsibility:** Real-time visualization and monitoring via WebSocket-connected frontend.

#### Component 6: Evaluation Engine
**Responsibility:** Compare TGNN against baselines and produce academic-quality evaluation results including plots, tables, and statistical significance tests.

---

## 6. Data Layer — Synthetic UPI Transaction Generator

Since no public UPI mule dataset exists, the project includes a purpose-built synthetic data generator. This is a core deliverable, not a stopgap. The generator must produce data realistic enough that patterns are non-trivial to detect and that model behavior would plausibly transfer to real data.

### 6.1 Legitimate Account Behavior Model

The generator first creates a population of legitimate accounts and simulates their transaction behavior over the time span of the dataset.

#### 6.1.1 Account Population

| Parameter | Distribution | Notes |
|---|---|---|
| Total accounts | 10,000 (train), 5,000 (val), 5,000 (test) | Distinct non-overlapping sets |
| Account age | Uniform(1, 730) days | Up to 2 years old |
| KYC type | 85% Full KYC, 15% Min KYC | Min KYC has lower limits |
| Bank | Weighted by real UPI market share | HDFC 25%, SBI 20%, ICICI 15%, Axis 10%, others 30% |
| VPA format | `name@ok{bank}` | e.g., `john123@okhdfcbank` |
| Device count | 1 device: 70%, 2 devices: 25%, 3+: 5% | Most users have 1-2 devices |
| Activity level | 60% active (≥1 txn/day), 40% casual (<1 txn/day) | Bimodal |

#### 6.1.2 Transaction Behavior

| Parameter | Distribution | Notes |
|---|---|---|
| Daily transaction count | Poisson(λ=5) for active, Poisson(λ=0.5) for casual | Per account |
| Transaction amount | LogNormal(μ=4.2, σ=1.5) | Median ~₹66, 99th percentile ~₹15,000 |
| P2P vs P2M ratio | 40% P2P, 60% P2M | Merchant transactions are more frequent |
| Counterparty diversity | 3–15 regular counterparts | Most users transact with a small set |
| Active hours | Bimodal: 10am–2pm, 6pm–10pm | Few transactions 12am–6am |
| Device consistency | Same device as previous txn: 95% | Device changes are rare for legit users |
| Merchant categories (P2M) | Groceries 25%, Food 20%, Transport 15%, Bills 15%, Shopping 10%, Entertainment 10%, Other 5% | Weighted by real UPI P2M distribution |

#### 6.1.3 Social Network Structure

Legitimate accounts form a social network that reflects real-world money flow patterns:

- Each account has 3–15 regular counterparts (friends, family, merchants)
- The network has community structure (small-world): accounts in the same social circle transact with each other more
- Merchant accounts have high in-degree (many customers) but low out-degree
- Personal accounts have balanced in/out degree
- The network grows organically: new accounts are added over time, forming new connections

### 6.2 Mule Account Behavior Model

Mule networks are planted into the legitimate graph using the following patterns. Each pattern is designed to be realistic — based on observed real-world mule behavior — and challenging — not trivially detectable by simple rules.

#### Pattern 1: Fan-Out Mule Network
- 1 source account sends to 5–20 mule accounts within a 10-minute window
- Each mule account receives 1–3 transactions
- Amounts: ₹2,000–₹50,000 per transaction (within legit range but skewed higher)
- Mule accounts are dormant for 30+ days before activation
- After receiving, each mule forwards to a different account (not the same destination — this distinguishes fan-out from coordinated ring)
- The source account itself may be a compromised legit account (not labeled as mule)
- Subtlety: amounts are within the normal range for P2P; the pattern is in the graph structure and temporal burst

#### Pattern 2: Layering Chain
- A → B → C → D (3–5 hops)
- Each hop within 2–10 minutes of receiving
- Amounts decrease slightly at each hop (5–10% "commission" extracted at each step)
- All accounts in chain are mules
- The first account (A) receives from an external source (the fraudster's entry point)
- The last account (D) either cashes out or forwards to an aggregator
- Subtlety: each individual transaction looks normal; the pattern is in the rapid sequential forwarding

#### Pattern 3: Fan-In Aggregator
- 10–30 mule accounts each send to 1 aggregator account
- Aggregator is the final mule (or the fraudster's exit point)
- Transactions spread over 30–60 minutes to avoid velocity rules
- Amounts vary to avoid amount-clustering detection
- The mule accounts may have received funds from various sources (fan-out from different entry points)
- Subtlety: the aggregator could look like a legitimate merchant receiving many payments — the distinguishing factor is that the senders are mules

#### Pattern 4: Cyclic Flow
- A → B → C → A (2–5 node cycle)
- Money returns to origin within 1 hour
- Indicates wash-trade or circular laundering
- Amounts may increase slightly at each hop (to simulate "investment returns" camouflage)
- Subtlety: cycles are rare in legitimate UPI (unlike, say, roommates splitting rent and paying each other back)

#### Pattern 5: Burst-Dormant
- Account dormant for 60+ days
- Sudden burst: receives and forwards 5–15 transactions within 30 minutes
- Returns to dormancy after burst
- May repeat the pattern every 2–7 days (irregular intervals)
- Subtlety: the dormancy period is the key signal — a consistently active account with the same transaction count would not be suspicious

#### Pattern 6: Coordinated Mule Ring
- 5–20 mule accounts activate simultaneously (within same hour)
- All receive from the same source
- All forward to the same destination (or to a small set of destinations)
- Pattern repeats at irregular intervals (every 2–7 days)
- The source and destination may change between activations
- Subtlety: individually, each mule looks like a normal low-activity account; the coordination is only visible in the graph

#### Pattern 7: Device-Sharing Mule Cluster
- 10–50 mule accounts all accessed from the same device fingerprint
- Device is distinct from any device used by legitimate accounts
- Transactions may be spread out to avoid velocity rules
- The accounts may not transact with each other (no inter-mule edges)
- Subtlety: the shared device is the only signal — no temporal or amount pattern is suspicious
- This pattern tests whether the model can use device-sharing edges in the graph

#### Pattern 8: Amount-Clustering Attack
- 50+ accounts all transact at near-identical amounts (e.g., ₹211, ₹262)
- Spread across many merchants (50-ish requests per merchant to stay under per-merchant thresholds)
- All from the same device (but device ID rotates every few hundred transactions)
- Mimics bot-driven card testing or automated mule operation
- Subtlety: per-merchant view shows only a few failed transactions; the graph view shows thousands from one device

#### Pattern 9: Gaming/Betting Funnel
- Multiple accounts registered under the same residential address
- All processing transactions for gaming/betting proceeds
- Merchants may be categorized under "fabric" or "textile" as a cover
- High-value transactions from both cards and UPI
- Subtlety: the shared address and merchant category are the signals, not the transaction amounts

### 6.3 Dataset Scale

| Dataset | Transactions | Accounts | Mule Accounts | Mule Ratio | Time Span | Purpose |
|---|---|---|---|---|---|---|
| Training | 200,000 | 10,000 | 500 (5%) | 5% | 30 days | Model training |
| Validation | 50,000 | 5,000 | 250 (5%) | 5% | 7 days | Hyperparameter tuning |
| Test | 50,000 | 5,000 | 250 (5%) | 5% | 7 days | Final evaluation |
| Stress Test | 500,000 | 20,000 | 1,000 (5%) | 5% | 30 days | Throughput/latency |
| Demo | Streaming | ~500 | ~25 (5%) | 5% | Live | Live demonstration |

### 6.4 Feature Engineering

#### 6.4.1 Node (Account) Features

| Feature | Type | Description | Computation |
|---|---|---|---|
| `account_age_days` | float | Days since first transaction | Static at generation time |
| `total_txns_in` | int | Lifetime incoming transaction count | Incremented on each incoming txn |
| `total_txns_out` | int | Lifetime outgoing transaction count | Incremented on each outgoing txn |
| `total_amount_in` | float | Lifetime incoming amount sum | Accumulated |
| `total_amount_out` | float | Lifetime outgoing amount sum | Accumulated |
| `unique_counterparts_in` | int | Distinct senders to this account | Set-based tracking |
| `unique_counterparts_out` | int | Distinct receivers from this account | Set-based tracking |
| `avg_txn_amount_in` | float | Mean incoming amount | total_amount_in / total_txns_in |
| `avg_txn_amount_out` | float | Mean outgoing amount | total_amount_out / total_txns_out |
| `txn_velocity_1h` | int | Transactions in last 1 hour | Sliding window counter |
| `txn_velocity_24h` | int | Transactions in last 24 hours | Sliding window counter |
| `amount_velocity_1h` | float | Total amount in last 1 hour | Sliding window sum |
| `device_count` | int | Number of distinct devices used | Set-based tracking |
| `dormancy_days` | float | Days since last transaction before current activity | Computed on each new txn |
| `kyc_type` | categorical | Full KYC (1) or Min KYC (0) | Static |
| `is_merchant` | binary | Whether account is a merchant (P2M receiver) | Static |
| `bank_id` | categorical | Encoded bank identifier | Static |
| `in_degree` | int | Current graph in-degree | Updated on edge addition |
| `out_degree` | int | Current graph out-degree | Updated on edge addition |
| `fan_out_ratio_1h` | float | out_degree in last 1h / total out_degree | Computed from sliding window |
| `layering_score` | float | Fraction of received amount forwarded within 1h | Computed from sliding window |
| `pagerank` | float | Graph centrality (computed periodically) | NetworkX pagerank on active subgraph |
| `betweenness_centrality` | float | Computed periodically on active subgraph | NetworkX betweenness |

#### 6.4.2 Edge (Transaction) Features

| Feature | Type | Description |
|---|---|---|
| `amount` | float | Transaction amount in INR |
| `timestamp` | float | Unix epoch (seconds) |
| `time_delta_from_prev_txn` | float | Time since sender's previous transaction |
| `amount_ratio` | float | amount / sender's avg outgoing amount |
| `txn_type` | binary | P2P (0) or P2M (1) |
| `device_id_hash` | categorical | Hashed device identifier |
| `merchant_category` | categorical | Encoded merchant category (if P2M) |
| `is_first_interaction` | binary | Whether this is the first txn between this pair |

#### 6.4.3 Derived Graph Features (for baselines, not for TGN which learns them)

These features are computed for the traditional ML baselines (Logistic Regression, XGBoost) and the rule-based system. The TGN model learns these patterns implicitly from the graph structure and does not need them as explicit features.

- `in_degree` / `out_degree` — current graph degree
- `fan_out_ratio_1h` — out_degree in last 1h / total out_degree
- `fan_in_ratio_1h` — in_degree in last 1h / total in_degree
- `layering_score` — fraction of received amount forwarded within 1h
- `pagerank` — graph centrality (computed every 1 hour on the active subgraph)
- `betweenness_centrality` — computed periodically on the active subgraph
- `clustering_coefficient` — local clustering coefficient
- `weakly_connected_component_size` — size of the WCC containing the node

### 6.5 Data Generation Pseudocode

```python
def generate_dataset(config):
    accounts = generate_legitimate_accounts(config.num_accounts, config.bank_distribution)
    social_network = build_social_network(accounts, config.avg_counterparts)
    transactions = []
    
    # Generate legitimate transactions
    for day in range(config.time_span_days):
        for account in active_accounts(accounts, day):
            num_txns = sample_poisson(account.daily_lambda)
            for _ in range(num_txns):
                txn = generate_legitimate_txn(account, social_network, day)
                transactions.append(txn)
    
    # Plant mule networks
    mule_accounts = generate_mule_accounts(config.num_mules, config.mule_patterns)
    for pattern in config.mule_patterns:
        mule_txns = inject_mule_pattern(pattern, accounts, config.time_span_days)
        transactions.extend(mule_txns)
    
    # Sort by timestamp (temporal order is critical for TGN)
    transactions.sort(key=lambda t: t.timestamp)
    
    # Split into train/val/test by time
    train, val, test = temporal_split(transactions, config.split_ratios)
    
    return train, val, test, accounts, mule_accounts
```

---

## 7. Temporal Graph Neural Network — Model Design

### 7.1 Architecture: Temporal Graph Network (TGN)

TGN (Rossi et al., ICML 2020) is the primary architecture. It is chosen over TGAT for its persistent per-node memory, which captures long-range temporal dependencies critical for mule detection (e.g., a dormant account's history before the dormancy period is informative).

#### 7.1.1 TGN at a Glance

For each transaction (interaction) at time `t` from sender `i` to receiver `j`:

1. **Message Function:** Compute a message vector from the sender's and receiver's current memory states, the time delta, and edge features.
2. **Message Aggregation:** Aggregate all messages for each node involved in the current batch.
3. **Memory Update:** Update each node's memory state using a GRU that takes the aggregated messages and the previous memory state.
4. **Embedding Computation:** Compute the node's embedding using temporal graph attention over its temporal neighborhood (the graph attention module uses the node's memory + neighbors' memories + time deltas).
5. **Classification:** Feed the embedding through an MLP classifier to get a mule probability score.

#### 7.1.2 TGN Modules — Detailed

**Module 1: Message Function**

For each interaction (transaction) at time `t` from node `i` to node `j`:

```
m_ij = MLP_msg([s_i(t), s_j(t), Δt_i, Δt_j, edge_features])
```

Where:
- `s_i(t)`, `s_j(t)` — memory states of sender and receiver at time `t` (before this interaction)
- `Δt_i` — time since node `i`'s last interaction
- `Δt_j` — time since node `j`'s last interaction
- `edge_features` — [amount, txn_type, amount_ratio, is_first_interaction, device_id_embedding]
- `MLP_msg` — 2-layer MLP, input dim = 2*memory_dim + 2 + edge_dim, hidden dim = 128, output dim = 64

**Module 2: Message Aggregation**

Messages to a node are aggregated using mean aggregation:

```
M_i(t) = mean(m_ij for all interactions involving i at time t in the current batch)
```

Alternative aggregations (implement as configurable options):
- **Last aggregation:** most recent message only
- **RNN aggregation:** messages processed through an RNN

**Module 3: Memory Updater**

Each node's memory is updated using a GRU:

```
s_i(t) = GRU(M_i(t), s_i(t_prev))
```

Where `s_i(t_prev)` is the node's memory state before this batch of interactions, and `GRU` is a standard gated recurrent unit with hidden dimension = memory_dim (64).

**Module 4: Embedding Module (Graph Attention)**

To compute the embedding `z_i(t)` of node `i` at time `t`, we use temporal graph attention over its temporal neighborhood `N(i, t)`:

```
z_i(t) = MultiHeadAttention(s_i(t), {s_j(t), Δt_ij for j in N(i, t)})
```

The attention mechanism:
- For each neighbor `j` of node `i`, compute a query-key-value attention:
  - Query: `W_q * s_i(t)`
  - Key: `W_k * [s_j(t), TimeEncoding(Δt_ij)]`
  - Value: `W_v * [s_j(t), TimeEncoding(Δt_ij)]`
- TimeEncoding uses Bochner's theorem: `TimeEncoding(t) = [cos(ω_1 t), sin(ω_1 t), ..., cos(ω_d t), sin(ω_d t)]` where `ω_k` are learnable frequencies
- Multi-head attention: 4 heads, each dim 32, concatenated to 128
- The neighborhood `N(i, t)` consists of the node's most recent K interactions (K=20 by default)

**Module 5: Classifier**

```
p_mule(i) = σ(MLP_cls(z_i(t)))
```

Where `MLP_cls` is a 3-layer MLP (128 → 64 → 32 → 1) with ReLU activations and dropout (0.2), and σ is sigmoid.

#### 7.1.3 Training

- **Loss:** Binary cross-entropy (mule vs. legitimate)
- **Optimizer:** Adam, learning rate 0.0001, weight decay 1e-5
- **Batch size:** 200 interactions per batch
- **Negative sampling:** Not needed (we have ground-truth labels from synthetic data)
- **Early stopping:** Patience 5 epochs on validation AUC-PR
- **Epoch definition:** One full pass through the training transaction stream in temporal order
- **Memory reset:** Node memories are reset at the start of each epoch (TGN processes the full temporal sequence per epoch to learn temporal patterns)
- **Gradient clipping:** Max norm 1.0 (prevents GRU instability)
- **Learning rate scheduler:** ReduceLROnPlateau on validation loss, factor 0.5, patience 3

#### 7.1.4 Key Implementation Details

- **Library:** Custom TGN implementation on top of `torch_geometric` (more control than `pytorch_geometric_temporal`, and avoids library incompatibility risk). A fallback using `pytorch_geometric_temporal` is maintained.
- **GPU:** Training on a single GPU (Colab T4 or local). TGN is memory-efficient — the memory module is O(N_nodes × memory_dim), not O(N_edges)
- **Mini-batch processing:** Process interactions in temporal order, in batches of 200. Update memories incrementally. Compute embeddings for the nodes involved in each batch.
- **Node memory initialization:** Zero-initialized for new nodes. Updated from the first interaction.
- **Memory batching:** During training, process the full temporal sequence. During inference, process each incoming batch and update memory incrementally.

#### 7.1.5 Why TGN and Not Alternatives

| Model | Captures Structure | Captures Temporal Evolution | Persistent Memory | Inductive | Suitability |
|---|---|---|---|---|---|
| Rule-based | No (per-account) | No | No | N/A | Baseline only |
| Logistic Regression | No | No | No | Yes | Baseline only |
| XGBoost | Partial (via graph features) | Partial (via velocity features) | No | Yes | Baseline only |
| GCN | Yes | No | No | No (transductive) | Baseline |
| GraphSAGE | Yes | No | No | Yes | Baseline |
| TGAT | Yes | Yes (time encoding) | No | Yes | Baseline |
| **TGN** | **Yes** | **Yes** | **Yes (GRU)** | **Yes** | **Proposed approach** |

### 7.2 Baseline Models for Comparison

All baselines are trained and evaluated on the same dataset splits. Static GNN baselines use graph snapshots taken at fixed intervals (every 1 hour), while TGNN models process the continuous temporal stream.

| Model | Type | Description | Implementation |
|---|---|---|---|
| Rule-Based | Heuristic | Velocity + fan-out + dormancy rules with hand-tuned thresholds | Custom Python |
| Logistic Regression | Traditional ML | Tabular per-account features (node features only) | scikit-learn |
| XGBoost | Gradient Boosting | Tabular per-account features + derived graph features | xgboost |
| GCN | Static GNN | Snapshot graph, 2-layer GCN, trained on final graph state | PyG GCNConv |
| GraphSAGE | Static GNN | Snapshot graph, neighborhood sampling, 2 layers | PyG SAGEConv |
| TGAT | Temporal GNN | Temporal attention without persistent memory | Custom on PyG |

### 7.3 Ablation Studies

Ablation studies remove or modify components of the TGN model to quantify their contribution. This is critical for academic rigor — the examiner needs to understand not just that TGN works, but why each component matters.

| Ablation | What's Removed | Expected Impact | Why |
|---|---|---|---|
| No memory module | TGN → TGAT (remove GRU, use only attention) | Lower recall on burst-dormant pattern | Memory stores dormancy history; without it, burst looks like normal activity |
| No temporal encoding | Remove Δt from message function and attention | Lower recall on layering chains | Model can't distinguish rapid sequential hops from spread-out transactions |
| No edge features | Remove amount, txn_type from messages | Lower precision (less context) | Amount and type distinguish mule transfers from legitimate payments |
| No attention | Replace attention with mean aggregation in embedding | Lower precision on complex patterns | Attention weights important neighbors; mean dilutes the signal |
| Reduced memory dim | memory_dim 64 → 16 | Lower recall on all patterns | Less capacity to store interaction history |
| Reduced neighborhood | K=20 → K=5 in attention | Lower precision | Smaller neighborhood = less graph context |
| 1-layer vs 2-layer classifier | Simpler decision head | Minimal impact expected | The embedding already captures complexity |

### 7.4 Hyperparameter Search Space

For reproducibility and academic rigor, the hyperparameter search space is documented:

| Hyperparameter | Search Range | Default |
|---|---|---|
| memory_dim | {32, 64, 128} | 64 |
| embedding_dim | {64, 128, 256} | 128 |
| message_dim | {32, 64, 128} | 64 |
| attention_heads | {2, 4, 8} | 4 |
| dropout | {0.1, 0.2, 0.3} | 0.2 |
| learning_rate | {1e-4, 5e-4, 1e-3} | 1e-4 |
| batch_size | {100, 200, 500} | 200 |
| neighborhood_size K | {10, 20, 50} | 20 |
| classifier_layers | {[128,64,32,1], [128,32,1], [64,32,1]} | [128,64,32,1] |

Search method: Grid search over the most impactful parameters (memory_dim, attention_heads, learning_rate), with others at default. Report the best configuration and the sensitivity analysis.

---

## 8. Real-Time Inference Pipeline

### 8.1 Streaming Architecture

```
Transaction Generator (synthetic or replay)
        │
        ▼
   Stream Source (asyncio.Queue or local Kafka)
        │
        ▼
   Stream Consumer (Python async)
        │
        ├──▶ Update Temporal Graph (add edge, update node features, manage sliding window)
        ├──▶ Update TGN Memory (GRU update for affected nodes)
        ├──▶ Compute Embeddings (temporal attention for affected nodes)
        ├──▶ Classify (mule probability)
        ├──▶ Send to Guardian (decision)
        ├──▶ Push to Dashboard (WebSocket)
        └──▶ Log to SQLite (audit trail)
```

### 8.2 Latency Budget

| Stage | Budget | Implementation |
|---|---|---|
| Graph update | 5ms | In-memory adjacency list (Python dict) |
| Memory update (GRU) | 10ms | PyTorch, batched for affected nodes |
| Embedding computation | 50ms | Temporal attention over K-hop neighborhood |
| Classification | 2ms | MLP forward pass |
| Guardian decision | 1ms | Threshold comparison |
| Dashboard push | 5ms | WebSocket emit |
| **Total** | **< 80ms** | **Well under 500ms target** |

### 8.3 Throughput Design

- Consumer processes transactions sequentially in temporal order (required by TGN — memories must be updated in order)
- For higher throughput: shard by sender VPA hash, run multiple TGN instances, merge scores
- For the project: single-threaded consumer is sufficient (100 txns/sec target is easily met)
- Batch processing: accumulate 200 transactions, process as a batch (more GPU-efficient than one-by-one)

### 8.4 Graph Memory Management

- **Sliding window:** Keep only the last 24 hours of transactions in the active graph
- **Node pruning:** Remove nodes with no interactions in the last 24 hours (their memory state is archived to disk/Redis)
- **Memory refresh:** If a pruned node reappears, reload its last memory state from archive
- **Maximum active graph size:** ~50,000 nodes, ~500,000 edges (fits comfortably in 8GB RAM)
- **Edge eviction:** Edges older than 24 hours are removed from the adjacency list but their effect is preserved in node memory

### 8.5 Inference Pseudocode

```python
async def inference_loop(stream_source, graph, tgn_model, guardian, dashboard):
    batch = []
    async for txn in stream_source:
        batch.append(txn)
        if len(batch) >= BATCH_SIZE:
            # 1. Update graph
            for t in batch:
                graph.add_edge(t.sender, t.receiver, t)
                graph.update_node_features(t)
            graph.evict_old_edges(current_time - WINDOW_HOURS * 3600)
            
            # 2. Update TGN memory
            affected_nodes = set(t.sender for t in batch) | set(t.receiver for t in batch)
            messages = tgn_model.compute_messages(batch, graph)
            tgn_model.update_memory(affected_nodes, messages)
            
            # 3. Compute embeddings and classify
            for node in affected_nodes:
                embedding = tgn_model.compute_embedding(node, graph)
                score = tgn_model.classify(embedding)
                
                # 4. Guardian decision
                decision = guardian.decide(node, score, graph.get_metadata(node))
                
                # 5. Push to dashboard and log
                await dashboard.push_update(node, score, decision, graph.get_neighbors(node))
                guardian.log_decision(decision)
            
            batch = []
```

---

## 9. Guardian — Decision & Action Engine

### 9.1 Decision Logic

Guardian receives a mule probability score for each account and translates it into one of four actions, with thresholds adjusted for account age (new accounts get stricter thresholds because they have less history to establish legitimacy).

```python
def guardian_decide(account_id, mule_score, account_metadata):
    age_days = account_metadata['account_age_days']
    is_new = age_days < 7
    is_merchant = account_metadata.get('is_merchant', False)

    # New accounts get stricter thresholds (less history to prove legitimacy)
    if is_new:
        thresholds = {'flag': 0.2, 'restrict': 0.5, 'freeze': 0.8}
    else:
        thresholds = {'flag': 0.3, 'restrict': 0.7, 'freeze': 0.9}

    # Merchants get slightly higher freeze threshold (high txn volume is normal)
    if is_merchant and not is_new:
        thresholds['freeze'] = 0.92

    if mule_score >= thresholds['freeze']:
        action = 'FREEZE'
        # Block all transactions, send alert
    elif mule_score >= thresholds['restrict']:
        action = 'RESTRICT'
        # Allow incoming, block outgoing
    elif mule_score >= thresholds['flag']:
        action = 'FLAG'
        # Allow all, add to review queue
    else:
        action = 'ALLOW'

    return {
        'account_id': account_id,
        'action': action,
        'score': mule_score,
        'thresholds_used': thresholds,
        'timestamp': now(),
        'explanation': generate_explanation(account_id, mule_score, account_metadata)
    }
```

### 9.2 Explanation Generation

Guardian produces a human-readable explanation for each action using the TGNN's attention weights and node features. This is critical for the "human-in-the-loop" review process — a risk ops person needs to understand why the model flagged an account.

**Template-based explanations:**

- "Account {vpa} flagged: {attention_pct}% of attention weight is on {num_neighbors} accounts that sent transactions within a {time_window}-minute window (fan-in pattern). Account was dormant for {dormancy_days} days before activation."

- "Account {vpa} frozen: Layering chain detected — {forward_pct}% of received amount forwarded within {forward_minutes} minutes to {num_downstream} downstream accounts, which also have high mule scores (avg: {avg_downstream_score})."

- "Account {vpa} restricted: Device sharing detected — {num_accounts_on_device} accounts accessed from the same device. {num_mule_on_device} of those accounts are already flagged as mules."

- "Account {vpa} flagged: Coordinated ring pattern — {num_ring_accounts} accounts activated within the same hour, all received from the same source ({source_vpa}), all forwarded to the same destination ({dest_vpa})."

**For the project, explanations are generated via template-based string formatting** using the model's attention weights, node features, and graph topology. This is sufficient for the academic demo. An optional LLM-based explanation (using Claude API or a local LLM) can be added as a stretch goal for more natural language summaries.

### 9.3 Action Logging & Audit Trail

Every Guardian decision is logged to SQLite with:

| Field | Type | Description |
|---|---|---|
| `decision_id` | UUID | Unique decision identifier |
| `account_id` | string | VPA of the account |
| `timestamp` | datetime | Decision timestamp |
| `mule_score` | float | Model's probability output |
| `action` | string | ALLOW / FLAG / RESTRICT / FREEZE |
| `thresholds_used` | JSON | Threshold values applied |
| `triggering_txn_id` | string | Transaction that triggered the evaluation |
| `top_5_attention_neighbors` | JSON | Top-5 neighbors by attention weight with their scores |
| `explanation` | text | Human-readable explanation |
| `human_review_status` | string | pending / approved / overturned |
| `human_reviewer` | string | (empty until reviewed) |
| `human_review_timestamp` | datetime | (empty until reviewed) |

### 9.4 Alert Engine

When Guardian issues a FREEZE or RESTRICT action, an alert is generated:

- **Dashboard alert:** Pushed via WebSocket to the alert feed panel (red/orange color-coded)
- **Log alert:** Written to SQLite audit log
- **Optional: Email alert** via SMTP (can be configured for demo)
- **Optional: Slack webhook** (can be configured for demo)

---

## 10. Dashboard & Visualization

### 10.1 Technology Choices

| Option | Stack | Pros | Cons | When to use |
|---|---|---|---|---|
| **Option A (Recommended)** | React + D3.js + TailwindCSS + FastAPI WebSocket | Polished, professional, full control | More dev time | Final demo, full marks on presentation |
| **Option B (Faster)** | Streamlit + pyvis/networkx + plotly | Quick to build, Python-only | Less polished, limited interactivity | MVP, early demo, time-constrained |

**Recommendation:** Start with Option B (Streamlit) for the MVP (Weeks 10–11), then upgrade to Option A (React + D3) for the final demo (Weeks 12–13). This de-risks the project — you have a working dashboard early, then polish it.

### 10.2 Dashboard Panels

#### Panel 1: Live Transaction Graph
- Force-directed graph layout, updated in real time
- Nodes sized by transaction volume (larger = more active)
- Nodes colored by risk score: green (< 0.3), yellow (0.3–0.7), orange (0.7–0.9), red (≥ 0.9)
- Edges animated as transactions flow (dash animation along edge direction)
- Flagged mule subgraphs highlighted with a semi-transparent bounding box
- Zoom/pan/filter controls (filter by time range, risk level, mule pattern type)
- Click any node → opens Account Inspector panel
- Performance: render on HTML5 Canvas (not SVG) for 1,000+ nodes at 30+ FPS
- Layout: incremental force-directed (don't re-layout the entire graph on each transaction — only update affected nodes)

#### Panel 2: Alert Feed
- Streaming list (newest first) of Guardian actions
- Each alert shows: account VPA, action (FLAG/RESTRICT/FREEZE), score, timestamp, pattern type detected, explanation
- Color-coded by severity: blue (FLAG), orange (RESTRICT), red (FREEZE)
- Click alert → highlights the account in the graph view and opens Account Inspector
- Auto-scroll with "pause" button to freeze the feed for reading

#### Panel 3: Metrics Dashboard
- Real-time line charts: precision, recall, F1, false-positive rate (rolling 1-hour window)
- Comparison bars: TGNN vs. Rule-based vs. Static GCN (side-by-side)
- Detection latency histogram (p50, p90, p95 markers)
- Confusion matrix (updated in real time, 2x2 grid with TP/TN/FP/FN counts)
- Throughput gauge: current transactions/second
- All charts use consistent styling (matplotlib-generated for report, D3/plotly for live dashboard)

#### Panel 4: Account Inspector (side panel)
- Selected account's VPA, age, KYC type, bank
- Risk score over time (line chart, last 24 hours)
- Transaction history (table: last 20 transactions with amount, counterparty, timestamp, type)
- TGN memory state visualization (heatmap of memory vector — shows what the model "remembers" about this account)
- Top-5 attention neighbors (list with weights — shows which accounts influenced the decision)
- Guardian explanation (formatted text)
- Action history for this account (list of past Guardian decisions)

### 10.3 WebSocket Protocol

```json
// Graph update (binary protocol for performance, JSON shown for clarity)
{
  "type": "graph_update",
  "node": {"id": "vpa_123", "score": 0.87, "action": "FREEZE", "size": 15},
  "edges": [{"from": "vpa_456", "to": "vpa_123", "amount": 5000, "timestamp": 1234567890}]
}

// Alert
{
  "type": "alert",
  "account_id": "vpa_123",
  "action": "FREEZE",
  "score": 0.87,
  "pattern": "layering_chain",
  "explanation": "Account frozen: 92% of received amount forwarded within 10 minutes...",
  "timestamp": 1234567890
}

// Metrics update (every 10 seconds)
{
  "type": "metrics",
  "precision": 0.88,
  "recall": 0.91,
  "f1": 0.89,
  "fpr": 0.025,
  "latency_p95": 72,
  "throughput": 115
}
```

### 10.4 Performance Requirements
- Render at 30+ FPS with up to 1,000 visible nodes
- Graph layout computed incrementally (not full re-layout on each transaction)
- WebSocket connection with binary protocol for graph updates (JSON for alerts)
- Throttle dashboard updates to 10/second (don't push every single transaction — batch them)

---

## 11. Evaluation Framework

### 11.1 Offline Evaluation (Training & Test)

**Dataset:** 300,000 total transactions across train/val/test splits (see Section 6.3)

**Protocol:**
1. Train all models on the training set (temporal order preserved for TGNN models)
2. Tune hyperparameters on validation set (grid search for top parameters, see Section 7.4)
3. Report final metrics on the held-out test set
4. Run ablation studies on the test set (Section 7.3)
5. Run per-pattern breakdown on the test set (Section 11.4)

**Metrics:**

| Metric | Formula | Why It Matters |
|---|---|---|
| Precision | TP / (TP + FP) | False accusations of legitimate users are costly and erode trust |
| Recall | TP / (TP + FN) | Missed mules continue to launder money — every missed mule is ongoing damage |
| F1 Score | 2 × P × R / (P + R) | Balanced view — the primary metric for comparison |
| AUC-ROC | Area under ROC curve | Threshold-independent ranking quality — does the model rank mules higher than non-mules? |
| AUC-PR | Area under PR curve | Better for imbalanced classes (5% mules) — more informative than AUC-ROC for this setting |
| False-Positive Rate | FP / (FP + TN) | Operational impact on legitimate users — high FPR means too many false alarms |
| Detection Latency | Time from mule's first fraudulent txn → flag | Speed of response — earlier detection = less money laundered |
| Throughput | Transactions processed per second | Scalability — can the system handle real-time load? |

### 11.2 Online Evaluation (Live Demo)

**Protocol:**
1. Start the live transaction stream (demo dataset, ~500 accounts, ~25 mules)
2. Dashboard shows the graph building in real time
3. TGNN scores update as transactions arrive
4. When planted mule networks activate, Guardian flags/freezes them
5. Record: detection rate, false positives, detection latency
6. Compare against rule-based baseline running in parallel on the same stream

### 11.3 Baseline Comparison Table (Expected Results)

| Model | Precision | Recall | F1 | FPR | Latency | AUC-ROC |
|---|---|---|---|---|---|---|
| Rule-Based | 0.65 | 0.55 | 0.60 | 0.08 | <10ms | 0.68 |
| Logistic Regression | 0.70 | 0.60 | 0.65 | 0.06 | <5ms | 0.72 |
| XGBoost + Graph Features | 0.78 | 0.72 | 0.75 | 0.04 | <10ms | 0.82 |
| GCN (static) | 0.80 | 0.70 | 0.75 | 0.04 | 100ms | 0.83 |
| GraphSAGE (static) | 0.82 | 0.73 | 0.77 | 0.035 | 80ms | 0.85 |
| TGAT | 0.85 | 0.80 | 0.82 | 0.03 | 60ms | 0.88 |
| **TGN (proposed)** | **0.88** | **0.90** | **0.89** | **0.025** | **<80ms** | **0.93** |

*Note: These are expected/target results, not measured. Actual results will be reported in the final report.*

### 11.4 Per-Pattern Evaluation

Report recall broken down by mule pattern type. This is the most insightful analysis — it shows where TGN's temporal memory provides the biggest advantage.

| Pattern | Rule-Based Recall | Static GCN Recall | TGN Recall (target) | TGN Advantage |
|---|---|---|---|---|
| Fan-Out | 0.70 | 0.80 | 0.95 | +0.15 over GCN |
| Layering Chain | 0.40 | 0.65 | 0.90 | +0.25 over GCN |
| Fan-In Aggregator | 0.65 | 0.75 | 0.92 | +0.17 over GCN |
| Cyclic Flow | 0.30 | 0.70 | 0.88 | +0.18 over GCN |
| Burst-Dormant | 0.50 | 0.55 | 0.93 | +0.38 over GCN |
| Coordinated Ring | 0.60 | 0.70 | 0.90 | +0.20 over GCN |
| Device-Sharing | 0.45 | 0.80 | 0.85 | +0.05 over GCN |
| Amount-Clustering | 0.55 | 0.75 | 0.88 | +0.13 over GCN |
| Gaming/Betting | 0.35 | 0.65 | 0.85 | +0.20 over GCN |

The key insight: temporal patterns (burst-dormant, layering chains) are where TGN should dramatically outperform static GNNs and rules. Device-sharing is where the advantage is smallest — it's primarily a structural pattern that static GNNs can also capture.

### 11.5 Statistical Significance

- Run each model 5 times with different random seeds (for data generation and model initialization)
- Report mean ± standard deviation for all metrics
- Paired t-test between TGN and each baseline (p < 0.05 considered significant)
- Report effect sizes (Cohen's d) for practical significance

### 11.6 Evaluation Plots for Report

Generate the following plots for inclusion in the final report (using matplotlib/seaborn with consistent styling):

1. **ROC curves** — all 7 models on one plot
2. **PR curves** — all 7 models on one plot (more informative than ROC for imbalanced data)
3. **Confusion matrices** — heatmap for TGN and for the best baseline
4. **Per-pattern bar chart** — recall by pattern type for Rule-based, GCN, TGN
5. **Ablation bar chart** — F1 for each ablation configuration
6. **Latency histogram** — distribution of inference latencies
7. **Training loss curve** — TGN training and validation loss over epochs
8. **Attention weight visualization** — for a sample mule account, show which neighbors had highest attention
9. **Graph visualization** — screenshot of the dashboard showing a detected mule ring
10. **Throughput vs. latency** — scatter plot showing the trade-off

---

## 12. Tech Stack

### 12.1 Complete Stack — Fully Open Source, Zero Proprietary Dependencies

| Layer | Technology | Why | Version |
|---|---|---|---|
| Language | Python 3.11 | Primary ML ecosystem, async support for streaming | 3.11+ |
| Deep Learning | PyTorch 2.x | Industry standard, best GNN support | 2.1+ |
| Graph Neural Networks | PyTorch Geometric (PyG) | TGN/TGAT implementation, inductive learning | 2.4+ |
| Graph Processing | NetworkX | Graph algorithms (pagerank, centrality) for features | 3.x |
| Traditional ML | scikit-learn, XGBoost | Baselines (LogReg, XGBoost) | 1.4+, 2.0+ |
| Data Processing | pandas, NumPy | Data prep, manipulation | 2.x, 1.26+ |
| Streaming | Python `asyncio.Queue` (MVP) or local Apache Kafka (docker-compose) | Transaction stream | N/A |
| Backend API | FastAPI | WebSocket for dashboard, REST for queries | 0.104+ |
| Frontend | React 18 + D3.js v7 + TailwindCSS (Option A) or Streamlit (Option B) | Real-time dashboard | 18.x, 7.x |
| Visualization | D3.js (graph), plotly (charts), matplotlib (report plots) | Graph viz, charts, publication-quality plots | 7.x, 5.x, 3.8+ |
| Database | SQLite | Audit logs, decisions, metrics — zero infra needed | built-in |
| Cache | Redis (docker-compose) or in-memory dict | Real-time graph state, node memory cache | 7.x |
| Model Tracking | MLflow (local) or Weights & Biases (free tier) | Experiment tracking, hyperparameter logging | 2.x |
| Version Control | Git + GitHub | Mandatory for academic project | N/A |
| Testing | pytest | Unit + integration tests | 7.x |
| Code Quality | ruff (linting) + black (formatting) | Automated code quality | 0.1+, 23.x |
| CI/CD | GitHub Actions | Automated tests on push | N/A |
| Deployment | Docker + docker-compose | One-command setup, reproducible environment | 24.x |
| Documentation | MkDocs | API docs, architecture docs | 1.x |
| Presentation | Marp (Markdown → slides) or PowerPoint | Presentation slides | N/A |

### 12.2 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU (training) | Google Colab T4 (free) | Colab A100 / local RTX 3060+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 20 GB (datasets, model checkpoints, plots) |
| CPU (inference) | 4 cores | 8 cores |
| Internet | Required for Colab, pip install | Broadband for GitHub Actions |

### 12.3 Cost Estimate

| Item | Cost |
|---|---|
| Google Colab (free tier) | ₹0 |
| GitHub (free tier) | ₹0 |
| Docker | ₹0 (Community Edition) |
| Redis, SQLite, all Python libraries | ₹0 (open source) |
| Domain name (optional, for dashboard) | ~₹500/year |
| **Total** | **₹0 – ₹500** |

This project is designed to be completely free to build and run.

---

## 13. Project Structure & Repository Layout

```
muleguard/
├── README.md                           # Setup, run, reproduce results
├── docker-compose.yml                  # One-command deployment
├── Dockerfile                          # Python + PyTorch + PyG
├── requirements.txt                    # Pinned dependencies
├── setup.py                            # Installable package
├── pyproject.toml                      # Modern Python project config
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Run tests on push
│       └── lint.yml                    # Lint + format check
├── config/
│   ├── model_config.yaml               # TGN hyperparameters
│   ├── data_config.yaml                # Data generation parameters
│   ├── guardian_config.yaml            # Decision thresholds
│   └── dashboard_config.yaml           # Dashboard settings
├── data/
│   ├── raw/                            # Generated raw transaction data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/                      # Graph snapshots, feature tensors
│   └── labels/                         # Ground-truth mule labels
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py                # Synthetic UPI transaction generator
│   │   ├── mule_patterns.py            # 9 mule pattern injectors
│   │   ├── preprocessor.py             # Feature engineering, graph construction
│   │   ├── dataset.py                  # PyTorch Dataset/DataLoader for TGN
│   │   └── vpa_generator.py            # Realistic VPA generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tgn.py                      # TGN model (5 modules)
│   │   ├── tgat.py                     # TGAT baseline
│   │   ├── gcn.py                      # Static GCN baseline
│   │   ├── graphsage.py                # Static GraphSAGE baseline
│   │   ├── xgboost_model.py            # XGBoost baseline
│   │   ├── logistic_regression.py      # LogReg baseline
│   │   ├── rule_based.py               # Rule-based baseline
│   │   └── modules/                    # Shared NN modules
│   │       ├── memory.py               # GRU memory module
│   │       ├── attention.py            # Temporal graph attention
│   │       ├── message.py              # Message function MLP
│   │       └── classifier.py           # Classification head
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── graph_builder.py            # Real-time temporal graph construction
│   │   ├── inference.py                # Real-time TGN inference engine
│   │   ├── stream_consumer.py          # asyncio.Queue / Kafka consumer
│   │   └── stream_producer.py          # Data generator → stream
│   ├── guardian/
│   │   ├── __init__.py
│   │   ├── decision.py                 # Threshold-based decision logic
│   │   ├── actions.py                  # Freeze/restrict/flag execution
│   │   ├── explanation.py              # Attention-weight-based explanations
│   │   ├── audit.py                    # SQLite action logging
│   │   └── alert.py                    # Alert engine (dashboard, log, optional email)
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI backend
│   │   ├── websocket.py                # WebSocket handler for live updates
│   │   ├── routes.py                   # REST API routes
│   │   └── static/                     # Frontend assets
│   │       ├── index.html              # Main dashboard page
│   │       ├── graph.js                # D3.js force-directed graph
│   │       ├── alerts.js               # Alert feed component
│   │       ├── metrics.js              # Metrics charts
│   │       └── inspector.js            # Account inspector panel
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Precision, recall, F1, AUC, latency
│   │   ├── comparison.py               # Baseline comparison runner
│   │   ├── ablation.py                 # Ablation study runner
│   │   ├── per_pattern.py              # Per-pattern breakdown analysis
│   │   ├── statistical.py              # Statistical significance tests
│   │   └── visualize.py                # Generate plots for report
│   └── utils/
│       ├── __init__.py
│       ├── config.py                   # YAML config loader
│       ├── logging.py                  # Structured logging
│       ├── seed.py                     # Reproducibility utilities
│       └── timer.py                    # Latency profiling
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: visualize transaction patterns
│   ├── 02_model_training.ipynb         # TGN training walkthrough
│   ├── 03_baseline_comparison.ipynb    # Compare all baselines
│   ├── 04_ablation.ipynb               # Ablation study
│   └── 05_visualization.ipynb          # Generate report plots
├── tests/
│   ├── test_generator.py               # Data generator tests
│   ├── test_mule_patterns.py           # Mule pattern injection tests
│   ├── test_graph_builder.py           # Graph construction tests
│   ├── test_tgn.py                     # TGN model tests
│   ├── test_tgat.py                    # TGAT baseline tests
│   ├── test_guardian.py                # Decision engine tests
│   ├── test_pipeline.py                # Integration tests
│   ├── test_dashboard.py               # Dashboard API tests
│   └── test_evaluation.py              # Metrics computation tests
├── scripts/
│   ├── generate_data.py                # Generate all datasets
│   ├── train_model.py                  # Train TGN model
│   ├── train_baselines.py              # Train all baselines
│   ├── run_baselines.py                # Evaluate all baselines
│   ├── run_ablation.py                 # Run ablation studies
│   ├── run_demo.py                     # Start live demo (stream + dashboard)
│   ├── evaluate.py                     # Full evaluation suite
│   └── generate_plots.py               # Generate all report plots
├── docs/
│   ├── architecture.md                 # System design documentation
│   ├── api_reference.md                # REST + WebSocket API docs
│   ├── data_schema.md                  # Transaction event schema
│   └── report/                         # Final project report
│       ├── main.tex                    # LaTeX source (IEEE format)
│       ├── references.bib              # Bibliography
│       └── figures/                    # Generated plots and diagrams
├── results/
│   ├── checkpoints/                    # Model checkpoints (.pt files)
│   ├── plots/                          # Generated evaluation plots
│   ├── tables/                         # Results tables (CSV/LaTeX)
│   └── logs/                           # Training and evaluation logs
└── .env.example                        # Example environment variables
```

---

## 14. Project Milestones & Timeline

### 16-Week Semester Plan

#### Phase 1: Foundation (Weeks 1–4)

**Week 1–2: Literature Review & Setup**
- Read 10+ papers on GNNs, TGNNs, fraud detection, mule account detection
- Write literature review section (5+ pages)
- Set up project repo, Docker environment, CI pipeline
- Install and test PyTorch Geometric, PyTorch
- Create project structure (all directories, `__init__.py` files)
- Write `requirements.txt` and `Dockerfile`
- Verify GPU access (Colab T4 or local)
- **Deliverable:** Literature review draft, project repo with working environment, `docker-compose up` works (empty dashboard)

**Week 3–4: Data Generator**
- Implement synthetic UPI transaction generator (`generator.py`)
- Implement VPA generator with realistic bank formats (`vpa_generator.py`)
- Implement all 9 mule pattern injectors (`mule_patterns.py`)
- Generate train/val/test datasets
- Write unit tests for data generation (`test_generator.py`, `test_mule_patterns.py`)
- Create EDA notebook: visualize transaction patterns, amount distributions, time-of-day patterns, mule vs. legit comparison (`01_data_exploration.ipynb`)
- Verify mule patterns are non-trivial: run simple rules and confirm they don't catch all mules
- **Deliverable:** Working data generator, datasets, EDA notebook, passing tests

#### Phase 2: Model Development (Weeks 5–9)

**Week 5–6: Baselines**
- Implement rule-based detector (`rule_based.py`) with velocity, fan-out, dormancy, device-sharing rules
- Implement Logistic Regression baseline (`logistic_regression.py`)
- Implement XGBoost baseline (`xgboost_model.py`)
- Implement static GCN baseline (`gcn.py`)
- Implement static GraphSAGE baseline (`graphsage.py`)
- Train and evaluate all baselines on the same dataset
- Record baseline metrics in `results/tables/`
- Write tests for each baseline model
- **Deliverable:** All 5 baselines trained, baseline metrics table, passing tests

**Week 7–9: TGN Implementation & Training**
- Implement TGN modules: message function (`message.py`), memory updater (`memory.py`), temporal graph attention (`attention.py`), classifier (`classifier.py`)
- Implement TGN model assembling all modules (`tgn.py`)
- Implement training loop with temporal-order batching (`train_model.py`)
- Train on synthetic data — verify loss converges
- Hyperparameter tuning: grid search over memory_dim, attention_heads, learning_rate
- Implement TGAT baseline (`tgat.py`) — TGN without memory
- Run ablation studies (`run_ablation.py`)
- Create model training notebook (`02_model_training.ipynb`)
- Create baseline comparison notebook (`03_baseline_comparison.ipynb`)
- **Deliverable:** Trained TGN model, TGAT baseline, model comparison table, ablation results

#### Phase 3: System Integration (Weeks 10–13)

**Week 10–11: Real-Time Pipeline**
- Implement temporal graph builder (`graph_builder.py`) — in-memory dynamic graph with sliding window
- Implement real-time inference engine (`inference.py`) — stream consumer → graph update → TGN inference → classification
- Implement stream producer and consumer (`stream_producer.py`, `stream_consumer.py`)
- Latency profiling and optimization (profile each stage, optimize hot paths)
- Throughput testing (verify 100+ txns/sec)
- Write integration tests (`test_pipeline.py`)
- **Deliverable:** Working real-time inference pipeline, latency/throughput benchmarks

**Week 12–13: Guardian + Dashboard**
- Implement Guardian decision engine (`decision.py`, `actions.py`, `explanation.py`, `audit.py`, `alert.py`)
- Implement action logging to SQLite
- Build dashboard backend (FastAPI + WebSocket) (`app.py`, `websocket.py`, `routes.py`)
- Build dashboard frontend:
  - Start with Streamlit MVP (graph viz via pyvis, alerts, basic metrics)
  - Then upgrade to React + D3.js (force-directed graph, alert feed, metrics charts, account inspector)
- Integrate pipeline → Guardian → dashboard end-to-end (`run_demo.py`)
- Test the live demo flow: start stream → graph builds → mules detected → alerts fire → metrics update
- Write dashboard tests (`test_dashboard.py`)
- **Deliverable:** Complete end-to-end system, working live demo

#### Phase 4: Evaluation & Polish (Weeks 14–16)

**Week 14: Evaluation**
- Run full evaluation (all 7 models, all 8 metrics) (`evaluate.py`)
- Run ablation studies (all 7 ablations)
- Per-pattern breakdown analysis
- Statistical significance tests (5 runs each, paired t-tests)
- Generate all evaluation plots (`generate_plots.py`)
- Create evaluation notebook (`04_ablation.ipynb`, `05_visualization.ipynb`)
- **Deliverable:** Complete evaluation results, all plots and tables

**Week 15: Demo & Report**
- Prepare live demo: test on a clean machine, rehearse 3+ times
- Write final project report (IEEE format, 20+ pages) in LaTeX
- Write API documentation (`api_reference.md`)
- Write architecture documentation (`architecture.md`)
- Write README with setup instructions, how to run, how to reproduce results
- Prepare presentation slides (15–20 slides)
- Record demo video (5 minutes, as backup)
- **Deliverable:** Final report, presentation, demo-ready system, demo video

**Week 16: Review & Submit**
- Internal review: test all components on a fresh clone, fix bugs
- Final code cleanup: remove dead code, ensure all tests pass, update docs
- Ensure `docker-compose up` starts everything on a clean machine
- Submit report, code, and demo video as required
- Present to examination panel
- **Deliverable:** Submitted project, presented demo

---

## 15. Deliverables

### 15.1 Code Deliverables

| # | Deliverable | Description |
|---|---|---|
| 1 | Source Code Repository | Complete, documented, version-controlled codebase on GitHub |
| 2 | Synthetic Data Generator | Standalone module that generates UPI transaction datasets with 9 mule patterns |
| 3 | TGN Model | Trained model with saved checkpoints and training scripts |
| 4 | Baseline Models | All 6 baselines (Rule-based, LogReg, XGBoost, GCN, GraphSAGE, TGAT) trained and evaluated |
| 5 | Real-Time Pipeline | Streaming inference engine with graph builder and TGN inference |
| 6 | Guardian Engine | Decision engine with thresholds, actions, explanations, and audit trail |
| 7 | Dashboard | Live visualization dashboard (React + D3.js or Streamlit) |
| 8 | Evaluation Suite | Metrics, comparison, ablation, per-pattern, statistical significance scripts |
| 9 | Docker Setup | `docker-compose.yml` for one-command deployment |
| 10 | Test Suite | Unit tests + integration tests (≥ 80% coverage) |
| 11 | CI/CD Pipeline | GitHub Actions for automated testing and linting |

### 15.2 Documentation Deliverables

| # | Deliverable | Description |
|---|---|---|
| 1 | Project Report | IEEE-format, 20+ pages: abstract, intro, lit review, methodology, architecture, implementation, evaluation, conclusion, references |
| 2 | Architecture Document | System design, component diagram, data flow, tech choices, UML diagrams |
| 3 | API Documentation | REST + WebSocket API reference (MkDocs) |
| 4 | README | Setup instructions, how to run, how to reproduce results |
| 5 | Presentation Slides | 15–20 slides for final presentation |
| 6 | Demo Video | 5-minute recorded demo (backup for live demo) |

### 15.3 Academic Deliverables

| # | Deliverable | Description |
|---|---|---|
| 1 | Problem Definition Document | Formal problem statement, scope, objectives |
| 2 | Literature Survey | 10+ papers reviewed, gap analysis, comparison table |
| 3 | Design Document | UML diagrams (use case, class, sequence, deployment), architecture, data design |
| 4 | Implementation Report | Code organization, key algorithms, implementation details |
| 5 | Testing Report | Test cases, results, coverage report |
| 6 | Results & Analysis | Evaluation tables, plots, statistical tests, per-pattern analysis |
| 7 | Conclusion & Future Work | Summary, limitations, roadmap |

---

## 16. Academic Requirements Mapping

### 16.1 Typical Major Project Evaluation Rubric

| Evaluation Criterion | Weight | How This Project Addresses It |
|---|---|---|
| **Problem Definition & Relevance** | 10% | UPI fraud is a real, large-scale problem with societal impact. Clearly motivated with data (500M+ daily UPI txns, RBI mandates, I4C advisories). The RBI Harbinger 2024 challenge explicitly included mule account detection as a problem statement. |
| **Literature Survey** | 10% | Comprehensive review of GNNs → TGNNs → fraud detection applications. 10+ papers reviewed with explicit gap analysis showing novelty. Comparison table: what each paper does vs. what this project does. |
| **System Design & Architecture** | 15% | Multi-component architecture with clear data flow. UML diagrams (use case, class, sequence, deployment), component diagram, tech stack justification. Each design decision documented with alternatives considered. |
| **Technical Complexity & Innovation** | 20% | TGNNs are cutting-edge. Combining temporal graph learning with real-time streaming and a decision engine is significantly beyond standard final-year projects. 9 mule patterns, 7 baselines, 7 ablations, custom TGN implementation. |
| **Implementation Quality** | 20% | Modular code with clean separation of concerns. Unit tests (≥80% coverage), type hints, docstrings, config files (no hardcoded paths), Docker, CI/CD. The repo should look like a real engineering project, not a notebook. |
| **Testing & Evaluation** | 15% | 7 models compared, 8 metrics, 7 ablations, 9 per-pattern breakdowns, statistical significance tests (5 runs, paired t-test). Publication-grade evaluation with proper plots. |
| **Report & Documentation** | 5% | IEEE-format report, 20+ pages. API docs (MkDocs), architecture doc, README. Proper citations (BibTeX). Clear figures with captions. |
| **Presentation & Demo** | 5% | Live dashboard demo with real-time graph visualization and mule detection. Visually compelling. Backup video recorded. Runs on examiner's machine via Docker. |

### 16.2 How to Maximize Marks in Each Category

**Problem Definition (10%):** Start the report with UPI transaction volume statistics (500M+ daily), NPCI/RBI reports on fraud growth, news articles on mule account crackdowns, the RBI's Harbinger 2024 challenge including mule detection as a problem statement. Show this is a real, unsolved, regulator-mandated problem.

**Literature Survey (10%):** Review at minimum: Kipf & Welling (GCN, ICLR 2017), Hamilton et al. (GraphSAGE, NeurIPS 2017), Veličković et al. (GAT, ICLR 2018), Rossi et al. (TGN, ICML 2020), Xu et al. (TGAT, ICLR 2020), Kumar et al. (JODIE, KDD 2019), plus 3-4 fraud detection papers using GNNs (Wang et al. 2019, Dou et al. 2020, Liu et al. 2021). Create a comparison table showing what each paper does vs. what this project does. Explicitly state the gap: no published work applies TGNNs to UPI mule detection with real-time inference.

**System Design (15%):** Draw proper UML — use case diagram (actors: data generator, stream consumer, TGN model, Guardian, dashboard user), class diagram (all major classes and relationships), sequence diagram (transaction arrival → graph update → TGN inference → Guardian decision → dashboard update), deployment diagram (Docker containers). Include the architecture diagram from this PRD. Justify each tech choice with alternatives considered.

**Technical Complexity (20%):** This is where the TGNN shines. The examiner should see: (1) deep learning model design with custom architecture (5 TGN modules), (2) real-time streaming pipeline, (3) graph algorithms, (4) full-stack dashboard. This is genuinely 3 projects' worth of complexity. The 9 mule patterns and 7 baselines show breadth; the 7 ablation studies show depth.

**Implementation (20%):** Clean code, proper naming, docstrings on every function, type hints, config files (no hardcoded paths), modular imports, `requirements.txt`, Docker. Run `pytest` in CI. Use `ruff` for linting and `black` for formatting. The repo should look like a real engineering project, not a notebook. Commit history should be clean and incremental.

**Testing & Evaluation (15%):** The 7-model comparison + 7 ablation studies + 9 per-pattern breakdowns + statistical significance tests is publication-grade evaluation. Generate proper plots (not screenshots): ROC curves, PR curves, confusion matrices, latency histograms, per-pattern bar charts, ablation bar charts, training loss curves, attention weight visualizations. Use matplotlib/seaborn with consistent styling for the report.

**Report (5%):** Use IEEE conference paper format or the college's specified format. 20+ pages. Proper citations (BibTeX). No typos. Clear figures with captions and references in text. Structure: Abstract → Introduction → Literature Review → Methodology → System Architecture → Implementation → Evaluation → Discussion → Conclusion → References.

**Demo (5%):** The live graph visualization with mule accounts being caught in real time is the "wow" moment. Rehearse it 3+ times. Have a backup video. Make sure it runs on the examiner's machine (`docker-compose up`). Show: (1) graph building, (2) legitimate transactions flowing, (3) mule network activating, (4) Guardian catching it in real time, (5) alert appearing, (6) account inspector showing the explanation.

---

## 17. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TGN training instability (loss not converging) | Medium | High | Start with TGAT (simpler), add memory module incrementally. Use gradient clipping (max norm 1.0). Validate on small dataset first. Reduce learning rate if loss oscillates. |
| Synthetic data too easy (TGNN gets 99% recall trivially) | Medium | Medium | Tune mule patterns to be subtle — overlap amounts with legitimate ranges, add noise, vary pattern intensities, include camouflaged patterns that mimic legit behavior. Add Pattern 8 (amount-clustering) and Pattern 9 (gaming/betting) which are harder to detect. |
| Synthetic data too hard (TGNN can't learn patterns) | Low | High | Start with obvious patterns, make them subtler incrementally. Ensure patterns are structurally distinct from legitimate behavior. If a specific pattern has <50% recall, investigate whether the pattern is too subtle or the model needs tuning. |
| Dashboard performance lag with large graphs | Medium | Medium | Limit visible nodes to top-500 by activity. Use incremental layout updates. Render on Canvas, not SVG. Throttle WebSocket updates to 10/second. |
| Real-time pipeline can't keep up with stream | Low | Medium | Batch process (200 txns per batch). Profile and optimize hot paths. Use GPU for inference. Reduce neighborhood size K if embedding computation is the bottleneck. |
| PyG / PyTorch version incompatibility | Medium | High | Pin all versions in `requirements.txt`. Have a custom TGN implementation as fallback (not dependent on PyG Temporal). Test on Colab and local. |
| Scope creep | High | Medium | Stick to this PRD. New ideas go in "Future Work" section of the report, not in the implementation. If ahead of schedule, polish existing components rather than adding new ones. |
| Time management (16 weeks is tight) | High | High | Follow the milestone schedule strictly. Weekly self-review against milestones. Cut dashboard polish before cutting evaluation rigor. If behind by Week 12, use Streamlit instead of React for the dashboard. |
| Examiner doesn't understand GNNs | Medium | Low | The report and presentation should include a clear, intuitive explanation of GNNs and TGN. Use analogies (e.g., "each account has a memory of its past interactions, like a diary"). The dashboard visualization makes the concept tangible. |
| Docker doesn't work on examiner's machine | Low | Medium | Test on multiple machines. Have a non-Docker fallback (`pip install -r requirements.txt && python scripts/run_demo.py`). Record a demo video as backup. |

---

## 18. Future Work

### 18.1 Short-term (Post-Project)

- **Heterogeneous graph:** Model different node types (customers, merchants, banks, devices) and edge types (P2P, P2M, refunds, device-sharing) explicitly using HetGNN or R-GCN. This would enable the model to learn different message functions for different edge types.
- **Explainability:** SHAP values on graph predictions, attention weight visualization, subgraph-based explanations ("this account was flagged because of this 4-node subgraph pattern"). Integrate with a local LLM for natural language explanations.
- **Online learning:** Update the TGN model incrementally as new transactions arrive (continual learning) rather than periodic retraining. This would enable the model to adapt to new fraud patterns in real time.
- **Adversarial robustness:** Test against deliberate evasion (fraudster aware of the model, tries to structure transactions to avoid detection). Implement adversarial training where the model is trained against an adversarial pattern generator.
- **Federated learning:** Train the model across multiple banks/payment entities without sharing raw transaction data, preserving privacy while leveraging cross-institution patterns.

### 18.2 Long-term

- **Real data integration:** Partner with a bank or NPCI to validate on real (anonymized) UPI transaction data. This is the ultimate validation of the synthetic data approach — does a model trained on synthetic data transfer to real data?
- **Multi-modal features:** Incorporate device fingerprinting signals (screen resolution, installed apps, sensor data), geolocation, app behavior patterns as additional node features. In production, device fingerprint contributes ~44% of a fraud model's discriminative signal.
- **Cross-platform:** Extend to IMPS, NEFT, RTGS transaction networks — mules operate across payment rails, and a cross-rail graph would capture patterns invisible within a single rail.
- **Production deployment:** Kubernetes-based deployment with auto-scaling, monitoring (Prometheus/Grafana), and alerting integration. Real-time model serving with sub-50ms latency targets.
- **Regulatory compliance:** RBI/NPCI reporting integration, PMLA (Prevention of Money Laundering Act) suspicious transaction reports, CPFIR (Central Payments Fraud Information Registry) integration. Automated STR (Suspicious Transaction Report) generation.
- **MNRL integration:** Integrate with TRAI's Mobile Number Revocation List to screen accounts against known fraudulent mobile numbers — a real-world signal that complements the graph-based detection.
- **Consortium data sharing:** Implement privacy-preserving cross-institution fraud detection using federated learning or secure multi-party computation, enabling banks to share mule account intelligence without exposing raw data.

---

## 19. References

### 19.1 Core Papers

1. Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). *Temporal Graph Networks for Deep Learning on Dynamic Graphs.* ICML 2020 Workshop on Graph Representation Learning.
2. Xu, D., Chuan, R., Aggarwal, C., et al. (2020). *Inductive Representation Learning on Temporal Graphs.* ICLR 2020.
3. Kumar, S., Zhang, X., & Leskovec, J. (2019). *Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks.* KDD 2019 (JODIE).
4. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
5. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017 (GraphSAGE).
6. Veličković, P., Cucurull, G., Casanova, A., et al. (2018). *Graph Attention Networks.* ICLR 2018.

### 19.2 Fraud Detection with GNNs

7. Liu, Y., Ao, X., Qin, Z., et al. (2021). *Pick and Choose: A GNN-based Imbalanced Flow Predictor.* CIKM 2021.
8. Wang, S., et al. (2019). *Heterogeneous Graph Attention Network for Fraud Detection.* CIKM 2019.
9. Dou, Y., Liu, Z., Sun, L., et al. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters.* CIKM 2020.
10. Liu, Y., et al. (2020). *Heterogeneous SimGNN with Application to Fraud Detection.* SDM 2020.
11. Palmucci, S., et al. (2023). *Real-time Fraud Detection with Graph Neural Networks.* IEEE Access.

### 19.3 Mule Account / Money Laundering Detection

12. Weber, N., et al. (2019). *SkillSet protecting customers and organizations from fraud.* KDD 2019 (Industry Track).
13. Almeida, P., et al. (2021). *Anti-Money Laundering using Graph Neural Networks.* IEEE International Conference on Data Mining.
14. FATF (Financial Action Task Force). *Money Laundering Typologies Reports.* fatf-gafi.org
15. RBI. *Master Directions on Frauds – Classification and Reporting.* rbi.org.in

### 19.4 UPI / Indian Payments Context

16. NPCI. *UPI Product Information.* npci.org.in
17. RBI. *Annual Report on Fraud — FY 2024-25.* rbi.org.in
18. RBI. *Harbinger 2024 — Innovation Challenge.* rbi.org.in (Problem statements #3 and #4 include mule account detection)
19. I4C (Indian Cyber Crime Coordination Centre). *Mule Account Suspect Repository.* (referenced in industry discussions)
20. TRAI. *Mobile Number Revocation List (MNRL) — Mandate for Banks and Regulated Entities.* (March 2025 mandate)

### 19.5 Libraries & Tools

21. PyTorch Geometric — https://pyg.org
22. PyTorch Geometric Temporal — https://github.com/benedekrozemberczki/pytorch_geometric_temporal
23. Deep Graph Library (DGL) — https://www.dgl.ai
24. NetworkX — https://networkx.org
25. FastAPI — https://fastapi.tiangolo.com
26. D3.js — https://d3js.org

### 19.6 Additional Reading

27. Jure Leskovec. *Fraud Detection is a Graph Problem.* LinkedIn Post, 2024. (Prof. Leskovec on why graph methods are the most shared approach for fraud detection)
28. Neo4j. *Transaction Graph Demo: How to Uncover Fraud.* YouTube. (Visual introduction to graph-based fraud detection)
29. Block (Square). *Building a Resilient, Real-Time Fraud System.* YouTube. (Industry talk on real-time fraud infrastructure)
30. Curve. *Curve scales fraud prevention with BigQuery Graph network analysis.* Google Cloud Blog, 2026. (Industry case study on graph-based fraud detection at scale)

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| TGNN | Temporal Graph Neural Network — GNN that operates on dynamic graphs where edges arrive over time |
| TGN | Temporal Graph Network — specific TGNN architecture with per-node memory (Rossi et al. 2020) |
| TGAT | Temporal Graph Attention Network — TGNN using temporal attention without persistent memory |
| GCN | Graph Convolutional Network — static GNN using spectral-based convolution |
| GraphSAGE | Inductive representation learning on large graphs — static GNN with neighborhood sampling |
| GAT | Graph Attention Network — static GNN with attention-based message passing |
| GRU | Gated Recurrent Unit — recurrent neural network used for TGN memory updates |
| VPA | Virtual Payment Address — UPI identifier (name@bank) |
| NPCI | National Payments Corporation of India — operates UPI |
| Mule account | Account used to receive and forward illicit funds |
| Layering | Moving illicit money through multiple accounts to obscure origin |
| Fan-out | One source sending to many destinations |
| Fan-in | Many sources sending to one aggregator |
| Guardian | The decision engine in MuleGuard that translates model scores into actions |
| AUC-PR | Area Under the Precision-Recall Curve — threshold-independent metric for imbalanced data |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve |
| Sliding window | Time-based window of active data kept in memory (24 hours in this project) |
| MNRL | Mobile Number Revocation List — TRAI database of revoked mobile numbers |
| CPFIR | Central Payments Fraud Information Registry — RBI's fraud registry |
| I4C | Indian Cyber Crime Coordination Centre |
| KYC | Know Your Customer — identity verification process |
| P2P | Person to Person transaction |
| P2M | Person to Merchant transaction |
| STR | Suspicious Transaction Report — regulatory filing |
| PMLA | Prevention of Money Laundering Act |

---

## Appendix B: Configuration Files

### model_config.yaml
```yaml
model:
  type: tgn
  memory_dim: 64
  embedding_dim: 128
  message_dim: 64
  attention_heads: 4
  dropout: 0.2
  classifier_layers: [128, 64, 32, 1]
  neighborhood_size: 20  # K for attention
  
training:
  learning_rate: 0.0001
  weight_decay: 0.00001
  batch_size: 200
  epochs: 50
  patience: 5
  gradient_clip: 1.0
  lr_scheduler:
    type: ReduceLROnPlateau
    factor: 0.5
    patience: 3
    
data:
  train_path: data/raw/train/
  val_path: data/raw/val/
  test_path: data/raw/test/
  sliding_window_hours: 24
  
ablation:
  no_memory: false        # If true, disable GRU (TGN → TGAT)
  no_temporal_encoding: false  # If true, remove Δt from messages
  no_edge_features: false      # If true, remove amount/type from messages
  no_attention: false          # If true, replace attention with mean aggregation
  reduced_memory_dim: null     # If set, override memory_dim
  reduced_neighborhood: null   # If set, override K
```

### data_config.yaml
```yaml
generation:
  num_accounts_train: 10000
  num_accounts_val: 5000
  num_accounts_test: 5000
  mule_ratio: 0.05
  time_span_days_train: 30
  time_span_days_val: 7
  time_span_days_test: 7
  
legitimate:
  account_age_range: [1, 730]
  active_ratio: 0.6
  daily_lambda_active: 5
  daily_lambda_casual: 0.5
  amount_lognormal_mu: 4.2
  amount_lognormal_sigma: 1.5
  p2p_ratio: 0.4
  p2m_ratio: 0.6
  device_count_1: 0.70
  device_count_2: 0.25
  device_count_3plus: 0.05
  counterpart_range: [3, 15]
  
mule_patterns:
  - fan_out
  - layering_chain
  - fan_in
  - cyclic_flow
  - burst_dormant
  - coordinated_ring
  - device_sharing
  - amount_clustering
  - gaming_betting

banks:
  - {name: okhdfcbank, weight: 0.25}
  - {name: okaxis, weight: 0.15}
  - {name: okstatebank, weight: 0.20}
  - {name: okicici, weight: 0.15}
  - {name: okpunjab, weight: 0.05}
  - {name: ybl, weight: 0.05}
  - {name: okdbs, weight: 0.03}
  - {name: others, weight: 0.12}
```

### guardian_config.yaml
```yaml
thresholds:
  normal_account:
    flag: 0.3
    restrict: 0.7
    freeze: 0.9
  new_account:  # age < 7 days
    flag: 0.2
    restrict: 0.5
    freeze: 0.8
  merchant_account:
    flag: 0.3
    restrict: 0.7
    freeze: 0.92
    
actions:
  freeze:
    block_incoming: true
    block_outgoing: true
    send_alert: true
    alert_level: critical
  restrict:
    block_incoming: false
    block_outgoing: true
    send_alert: true
    alert_level: warning
  flag:
    block_incoming: false
    block_outgoing: false
    send_alert: false
    add_to_review_queue: true
    alert_level: info
    
alerts:
  dashboard: true
  sqlite_log: true
  email:
    enabled: false
    smtp_server: ""
    recipients: []
  slack:
    enabled: false
    webhook_url: ""
```

---

## Appendix C: Real-World Fraud Patterns Observed in Industry

This appendix documents real-world fraud patterns that informed the design of the synthetic mule patterns in this PRD. These patterns were identified through analysis of fraud detection discussions from fintech engineering teams.

### C.1 Device Fingerprint Integration for Mule Detection

**Pattern:** Coordinated and collusive chargeback abuse attacks through both first-party and mule accounts from certain geographies.

**Detection approach:** Device fingerprinting integrated on onboarding, login, and checkout pages to identify devices used to create fraudulent accounts. Setup alerts for MIDs created from the same device, merchant-customer collusion, and same devices logging into multiple MIDs.

**Impact:** 350+ fraudulent accounts identified. Time to detect reduced by ~49%, average GMV to detect reduced by ~90%. Money loss reduced by 92% in subsequent attacks, amounting to ~₹3 Cr reduction.

**Relevance to MuleGuard:** This validates the device-sharing edge in the graph. Pattern 7 (Device-Sharing Mule Cluster) in the synthetic generator is directly inspired by this.

### C.2 MNRL (Mobile Number Revocation List) Integration

**Pattern:** Fraudsters use fake documents to get mobile numbers, then exploit victims' KYC documents to create mule accounts.

**Detection approach:** Integration with TRAI's MNRL database of revoked mobile numbers. All existing merchants screened against revoked numbers.

**Impact:** 6,000+ matches on live merchants, 1,550 already risk-actioned, 400+ disabled. 250+ matches in banking, 140+ actioned.

**Relevance to MuleGuard:** This is a real-world signal that complements graph-based detection. In the synthetic data, we simulate this by having some mule accounts share phone numbers with known-fraudulent patterns.

### C.3 Cross-Merchant Fraud Ring Detection

**Pattern:** A single device making 47 failed payments across 12 different merchants in one hour. Per-merchant, only 3–4 failures are visible — below any per-merchant threshold. The coordinated pattern is only visible at the network level.

**Detection approach:** Knowledge graph of entities across all merchants (devices, IPs, cards, VPAs, bank accounts, merchants). Graph traversal in real time to identify multi-merchant fraud patterns.

**Relevance to MuleGuard:** This is exactly the type of pattern that the TGN model should learn — graph structure that's invisible per-merchant but obvious in the graph. Pattern 8 (Amount-Clustering Attack) simulates this.

### C.4 Card Testing Attack with Device Rotation

**Pattern:** A device sending 9,500 requests with 7,500 cards, 8,500 email addresses, and 7,100 IP addresses across 105 countries. 99.8% of transactions at exactly ₹211. At most ~50 requests per merchant to spread the load and avoid per-merchant detection.

**Detection approach:** Anomaly monitor with rules: >15 transactions per IP in 24h, >3 email addresses per device in 24h, >5 card hashes per IPv6 in 1h. But these are per-merchant rules — the coordinated pattern requires cross-merchant graph analysis.

**Relevance to MuleGuard:** Pattern 8 (Amount-Clustering Attack) is directly inspired by this. The near-identical amount (₹211) and device rotation are key features.

### C.5 Gaming/Betting Mule Funnel

**Pattern:** Multiple mule accounts registered under the same residential address, processing gaming/betting transaction proceeds. Accounts categorized under "fabric" or "textile" as a cover business.

**Detection approach:** Shared address detection across accounts, unusual transaction patterns for the declared business category, high-value transactions from both cards and UPI.

**Relevance to MuleGuard:** Pattern 9 (Gaming/Betting Funnel) is directly inspired by this. The shared address is a graph edge, and the category mismatch is a node feature.

### C.6 Layering Chain with Commission Extraction

**Pattern:** A → B → C → D, each hop within 2–10 minutes, with 5–10% "commission" extracted at each hop. All accounts are mules, often compromised legitimate accounts.

**Detection approach:** Velocity rules (fast forwarding), amount decrease detection (commission extraction). But these rules can be evaded by slowing down or varying amounts.

**Relevance to MuleGuard:** Pattern 2 (Layering Chain) simulates this. The TGN's temporal memory should capture the rapid sequential forwarding pattern even when individual transactions look normal.

### C.7 Anomaly in Transaction Pattern (Industry Risk Flag)

**Pattern:** "anomaly_in_transaction_pattern" is the most common risk reason code used by risk teams to suspend merchants. This is a catch-all for behavior that deviates from the account's established baseline.

**Detection approach:** Currently rule-based — velocity thresholds, amount thresholds, dormancy-then-activity rules. But these are reactive and require manual rule creation.

**Relevance to MuleGuard:** The TGN model learns what "normal" looks like for each account and flags anomalies structurally — without needing a human to write a rule for each new pattern. This is the core value proposition of the project.

### C.8 Fibe / TrustScan Device-Level Verification

**Pattern:** Fraudster gains access to a legitimate user's mobile number/account and obtains OTP. The mobile number may have a good historical risk profile, but the transaction is from an unknown device.

**Detection approach:** Device-level signals: new/unusual device activity, device-user consistency, device reputation, multiple accounts on same device, SIM/device change signals.

**Relevance to MuleGuard:** This validates the importance of device features as node attributes. The device-sharing edge is one of the strongest signals in the graph.

---

## Appendix D: Sample Transaction Event Schema

### D.1 Transaction Event (JSON)

```json
{
  "txn_id": "txn_20260921_000123",
  "sender_vpa": "arjun.kumar@okhdfcbank",
  "receiver_vpa": "merchant.flipkart@okaxis",
  "amount": 2499.00,
  "currency": "INR",
  "timestamp": 1726910400.0,
  "txn_type": "P2M",
  "device_id": "dev_a1b2c3d4e5f6",
  "device_type": "android",
  "merchant_category": "shopping",
  "sender_kyc_type": "full",
  "receiver_kyc_type": "full",
  "sender_account_age_days": 456,
  "receiver_account_age_days": 730,
  "is_mule_sender": false,
  "is_mule_receiver": false,
  "mule_pattern_type": null,
  "metadata": {
    "ip_address": "106.51.34.xxx",
    "geo_city": "Bengaluru",
    "geo_state": "Karnataka",
    "app_name": "PhonePe",
    "os_version": "Android 14"
  }
}
```

### D.2 Mule Transaction Event (Layering Chain)

```json
{
  "txn_id": "txn_20260921_000456",
  "sender_vpa": "mule_001@okhdfcbank",
  "receiver_vpa": "mule_002@okaxis",
  "amount": 4500.00,
  "currency": "INR",
  "timestamp": 1726910460.0,
  "txn_type": "P2P",
  "device_id": "dev_fraud_device_001",
  "device_type": "android",
  "merchant_category": null,
  "sender_kyc_type": "full",
  "receiver_kyc_type": "min",
  "sender_account_age_days": 92,
  "receiver_account_age_days": 15,
  "is_mule_sender": true,
  "is_mule_receiver": true,
  "mule_pattern_type": "layering_chain",
  "metadata": {
    "ip_address": "49.36.xxx.xxx",
    "geo_city": "Jaipur",
    "geo_state": "Rajasthan",
    "app_name": "BHIM",
    "os_version": "Android 12"
  }
}
```

### D.3 Guardian Decision Output (JSON)

```json
{
  "decision_id": "dec_20260921_000789",
  "account_id": "mule_002@okaxis",
  "timestamp": 1726910461.5,
  "mule_score": 0.94,
  "action": "FREEZE",
  "thresholds_used": {
    "flag": 0.3,
    "restrict": 0.7,
    "freeze": 0.9
  },
  "triggering_txn_id": "txn_20260921_000456",
  "top_5_attention_neighbors": [
    {"vpa": "mule_001@okhdfcbank", "attention_weight": 0.38, "mule_score": 0.96},
    {"vpa": "mule_003@okicici", "attention_weight": 0.25, "mule_score": 0.91},
    {"vpa": "source_unknown@ybl", "attention_weight": 0.18, "mule_score": 0.99},
    {"vpa": "mule_004@okstatebank", "attention_weight": 0.11, "mule_score": 0.87},
    {"vpa": "legit_user@okhdfcbank", "attention_weight": 0.08, "mule_score": 0.12}
  ],
  "explanation": "Account mule_002@okaxis frozen: Layering chain detected — 94% of received amount (₹4,500) forwarded within 6 minutes to 2 downstream accounts (mule_003, mule_004), which also have high mule scores (avg: 0.89). Account was dormant for 45 days before activation. Top attention neighbor mule_001@okhdfcbank (weight: 0.38) is a confirmed mule with score 0.96.",
  "human_review_status": "pending"
}
```

---

## Appendix E: Mathematical Formulation of TGN

### E.1 Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{G}(t)$ | Dynamic graph at time $t$ |
| $\mathcal{V}(t)$ | Set of nodes (accounts) active at time $t$ |
| $\mathcal{E}(t)$ | Set of edges (transactions) at time $t$ |
| $s_i(t)$ | Memory state of node $i$ at time $t$ |
| $z_i(t)$ | Embedding of node $i$ at time $t$ |
| $e_{ij}$ | Edge features for transaction from $i$ to $j$ |
| $\Delta t_i$ | Time since node $i$'s last interaction |
| $\mathcal{N}(i, t)$ | Temporal neighborhood of $i$ at time $t$ |

### E.2 Message Function

For an interaction from node $i$ to node $j$ at time $t$:

$$\mathbf{m}_{ij}(t) = \text{MLP}_{\text{msg}}\left([\mathbf{s}_i(t^-), \mathbf{s}_j(t^-), \Delta t_i, \Delta t_j, \mathbf{e}_{ij}]\right)$$

Where $t^-$ denotes the time just before the interaction, and $\text{MLP}_{\text{msg}}$ is a 2-layer MLP.

### E.3 Message Aggregation

$$\mathbf{M}_i(t) = \text{mean}\left(\{\mathbf{m}_{ij}(t) : \forall j \text{ interacting with } i \text{ at } t\}\right)$$

### E.4 Memory Update

$$\mathbf{s}_i(t) = \text{GRU}\left(\mathbf{M}_i(t), \mathbf{s}_i(t^-)\right)$$

### E.5 Temporal Graph Attention (Embedding)

For node $i$ at time $t$, with multi-head attention over its temporal neighborhood $\mathcal{N}(i, t)$:

$$\mathbf{z}_i(t) = \text{MultiHead}\left(\mathbf{s}_i(t), \{(\mathbf{s}_j(t), \Delta t_{ij}) : j \in \mathcal{N}(i, t)\}\right)$$

Where each attention head computes:

$$\text{head}_h = \text{softmax}\left(\frac{(\mathbf{W}_Q \mathbf{s}_i) \cdot (\mathbf{W}_K [\mathbf{s}_j, \phi(\Delta t_{ij})])^T}{\sqrt{d_k}}\right) (\mathbf{W}_V [\mathbf{s}_j, \phi(\Delta t_{ij})])$$

And $\phi(\Delta t)$ is the time encoding based on Bochner's theorem:

$$\phi(\Delta t) = [\cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \ldots, \cos(\omega_d \Delta t), \sin(\omega_d \Delta t)]$$

with learnable frequencies $\omega_1, \ldots, \omega_d$.

### E.6 Classification

$$p_{\text{mule}}(i) = \sigma\left(\text{MLP}_{\text{cls}}(\mathbf{z}_i(t))\right)$$

Where $\text{MLP}_{\text{cls}}$ is a 3-layer MLP and $\sigma$ is the sigmoid function.

### E.7 Training Loss

$$\mathcal{L} = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left[ y_i \log p_{\text{mule}}(i) + (1 - y_i) \log(1 - p_{\text{mule}}(i)) \right]$$

Where $\mathcal{B}$ is the current batch of nodes and $y_i \in \{0, 1\}$ is the ground-truth label (mule = 1).

---

**End of PRD — MuleGuard v2.0**

---

*This PRD is a living document. Version 2.0 incorporates insights from real-world fraud detection patterns observed in fintech engineering teams, including device fingerprinting, cross-merchant graph analysis, MNRL integration, card-testing attack signatures, gaming/betting mule funnels, and the regulatory emphasis on proactive mule account detection. The project is designed to be completely self-contained — buildable from scratch with zero proprietary dependencies, using only open-source tools, synthetic data, and free cloud resources.*
