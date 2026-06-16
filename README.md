# CRE-DTS-DeepSurv-Risk-Modeling
CRE-DTS model is a forward-looking framework for assessing CRE credit risk at U.S. financial institutions , with a particular focus on small- and mid-sized CRE lending and investment. 
Algorithmic Financial Salvage: Mitigating the 2026 Commercial Real Estate Maturity Wall through Dynamic Repositioning and Machine Learning
Author: Ho Fan Wu

## Abstract:
  The U.S. financial system faces a structural crisis in 2026 as more than $4 trillion in Commercial Real Estate (CRE) debt matures under a contractionary "higher-for-longer" interest rate environment. Traditional macroeconomic supervisory frameworks, such as Basel III, exhibit critical blind spots in detecting non-linear defaults driven by remote-work structural shifts and opaque private credit markets. This paper introduces the CRE-DTS (Commercial Real Estate Dynamic Time-to-Event System), an interdisciplinary framework integrating quantitative finance and survival-analysis machine learning. Rather than executing passive risk isolation, the proposed endeavor serves as an algorithmic blueprint for strategic asset repositioning—transforming distressed office spaces into critical digital infrastructure (data centers) and multi-family housing. This framework effectively bridges the gap between macroeconomic vulnerability and micro-level architectural resilience, safeguarding regional banking liquidity and urban economic stability.
  
## 1. Introduction and Background:
   
  The 2026 Macro-Financial CrisisAccording to the Federal Reserve’s May 2026 Financial Stability Report (pp. 17-21), the convergence of structural demand shifts in office spaces and compressed credit conditions has elevated commercial real estate (CRE) to a primary systemic vulnerability in the United States economy. Industry estimates indicate that total CRE loan maturities between 2025 and 2029 exceed $4 trillion, with a substantial portion originated during historical interest rate troughs of 3% to 4%. Refinancing these obligations in 2026 commonly occurs at rates exceeding 6% to 7%, creating an unprecedented "interest rate cliff" that threatens to destabilize regional banks, which hold nearly 70% of all domestic CRE debt.

  Simultaneously, the persistent rise of remote work has decoupled office property valuations from standard economic recovery cycles. This structural rupture yields a vast volume of "stranded assets"—underutilized metropolitan office buildings that face imminent debt service default due to degraded Highest and Best Use (HBU) dynamics.

## 2. Litigations of Standard Regulatory Frameworks (Basel III Blind Spots)
  Traditional banking regulations codified under the Basel III framework rely heavily on static, historical capital adequacy metrics and linear probability-of-default (PD) models. In the current 2026 macroeconomic landscape, these legacy frameworks manifest three critical data blind spots:

### 2.1. Risk Diffusion via Nonbank Financial Intermediaries (Shadow Banking)
  As outlined in the Federal Reserve's Financial Stability Report (pp. 32-35), an increasing share of CRE debt has migrated to nonbank financial institutions, private equity, and private credit markets. Basel III heavily regulates traditional deposit-taking institutions but fails to capture the risk propagation networks within these unregulated "shadow banks." The systemic opacity prevents regulators from tracing the hidden leverage connecting traditional banks to private distress.

### 2.2. The Cytokine Storm of Collective Credit Contraction
  When default rates escalate, Basel III dynamics inadvertently trigger a pro-cyclical risk mitigation mechanism. Banks collectively tighten lending standards and restrict liquidity (Federal Reserve, 2026, pp. 25-28). This indiscriminate credit contraction induces a financial "cytokine storm"—starving viable properties of refinancing capital, depressing broader property valuations, and converting localized liquidity constraints into a systemic solvency crisis.

## 3. Methodology: The CRE-DTS Machine Learning Architecture
To resolve these systemic blind spots, this paper proposes the CRE-DTS (Dynamic Time-to-Event System). By synthesizing quantitative financial engineering with survival analysis machine learning, the framework shifts the paradigm from binary default classification to temporal, adaptive intervention.3.1. Mathematical Formulation of Temporal Default WindowsInstead of modeling whether an asset will default ($PD \in [0,1]$), the system applies a machine-learning-driven Cox Proportional Hazards Model to calculate when an asset will default. The dynamic survival probability $S(t)$ over a time horizon $t$ is expressed as:

$$S(t) = P(T > t) = \exp\left( -\int_0^t \lambda_0(u) \exp(\beta^T X) du \right)$$

Where: 
 - $\lambda_0(u)$ represents the baseline hazard function under severe macroeconomic stress.
 - $X$ is a vector of high-dimensional alternative covariates, including real-time metropolitan mobility data, localized corporate downsizing velocity, and contractual lease expirations extracted via Natural Language Processing (NLP).
 - $\beta$ represents the learned non-linear weights of risk factors under high-interest-rate shocks.

### 3.2. Alternative Data Panopticon for Shadow Credit Tracking
To penetrate the shadow banking black box, the system bypasses standard voluntary institutional disclosures. By utilizing specialized data parsing algorithms developed via Python, the model ingests non-traditional indicators: municipal property title re-registrations, corporate leadership attrition rates on professional networks, and local commercial litigation filiations. This alternative data matrix exposes hidden distress signals 6 to 12 months before they manifest in official regulatory filings.


    [Alternative Data Input: NLP Legal Filings + Municipal Registries]
                             
                             │
                             ▼
              
              [CRE-DTS Survival Engine: S(t)]
                             
                             │
                             ▼
                             
      [Dynamic Temporal Risk Window Identification (6-12 Mo)]
                             
                             │
                             ▼
    
    [Strategic Asset Repositioning Strategy Selection (HBU Analysis)]
        
        ├──> Path A: Office-to-Data Center (STEM/Digital Infrastructure)
        
        └──> Path B: Office-to-Multifamily (Urban Housing Supply)


## 4. Strategic Asset Repositioning: Turning Liabilities into Infrastructure
The ultimate utility of the CRE-DTS framework lies in its predictive transition from risk identification to architectural salvage. Identifying a temporal default window allows regional banks and institutional investors to execute Strategic Asset Repositioning through Adaptive Reuse before a catastrophic foreclosure occurs.

### 4.1. Office-to-Data Center Conversion (Digital Infrastructure Re-alignment)
Stranded downtown office corridors possess core attributes highly valuable to specialized industrial sectors: robust floor-load capacities, high vertical clearance, and concentrated municipal grid connections. The model assesses the financial feasibility of transforming distressed assets into Edge Data Centers or high-density computing hubs. This conversion directly aligns with national strategic mandates to reinforce domestic technological infrastructure and expand computational capacity amidst the ongoing AI hardware expansion.

### 4.2. Office-to-Multifamily Housing Conversion (Socio-Economic Stabilization)
For assets located in regions suffering from dense residential supply deficits, the model executes a automated Highest and Best Use (HBU) Optimization Analysis. It evaluates whether retrofitting commercial envelopes into multi-family housing yields a superior Yield-on-Cost relative to liquidation. This structural transition mitigates systemic bank losses while simultaneously addressing the acute national housing shortage, returning vital tax bases to municipal governments, and preventing the urban decay typically induced by prolonged commercial vacancies.

## References
 - Board of Governors of the Federal Reserve System. (2026). Financial Stability Report. Washington, DC: Federal Reserve. pp. 17-21, 25-28, 32-35. Available at: https://www.federalreserve.gov/publications/files/financial-stability-report-20260508.pdf
 - Board of Governors of the Federal Reserve System. (2025). Financial Stability Report: Asset Valuations. Washington, DC: Federal Reserve. Available at: https://www.federalreserve.gov/publications/April-2025-financial-stability-report-Asset-Valuations.htm
 - MSCI Real Assets. (2025). Commercial Real Estate Maturity Wall Analysis and Liquidity Constraints. MSCI Research Bulletin.
 - Trepp, LLC. (2026). The CRE Maturity Wall: Analyzing the $4 Trillion Debt Refinancing Cliff. Trepp Insights.

# CRE-DTS — Address-Centric Data & Entity Resolution Layer

> Turning fragmented public real-estate records into a current, borrower-level
> view of commercial real estate (CRE) credit risk.

This document explains the **data ingestion and entity resolution layer** that
sits underneath the CRE-DTS survival model. It describes *what we are trying to
do*, *why the problem is hard*, and *how the pipeline solves it*, with runnable
example code.

---

## 1. The goal

Existing CRE data platforms are expensive and, more importantly, **backward-looking**.
A typical underwriting workflow pulls:

- a sale that recorded nine months ago,
- an owner of record that is two LLCs out of date,
- rent comps from a building that re-tenanted last quarter.

You end up paying premium prices for a *lagging snapshot*. The information needed
to do better is already public — county records, recorded deeds, permits, listing
feeds, satellite and street-level imagery. The hard part was never *access*. It was
**stitching everything into one current view, per address, fast enough to matter**.

Our goal is a system where you:

> **Type an address → get current ownership, last sale, recorded debt, comps
> inside a real radius, and a first-pass NOI — in seconds, not days.**

That single-address view then feeds a forward-looking credit risk score. This
repository builds the methodology end to end, and the data layer described here
is its foundation.

---

## 2. Why this is hard: the three real problems

The challenge is not downloading data. It is reconciling it.

### 2.1 Entity resolution (the hardest part)

County deeds list owners as LLCs and single-purpose entities (SPEs). The same
beneficial owner appears under many entity names over time:

```
CENTRAL TOWER OWNER LLC
   → CENTRAL TOWER OWNER, L.L.C.        (correction deed, formatting change)
   → DESERT CAP CENTRAL TOWER LLC       (renamed entity, looks unrelated)
```

To get a borrower-level view of exposure, these scattered records must be
linked back to a single beneficial owner. This is a **record-linkage** problem,
and it is where most of the analytical value lives.

### 2.2 Address as an unstable key

`"450 W Camelback Rd"`, `"450 West Camelback Road, Ste 100"`, and a typo'd
variant all refer to the same asset. Raw address strings are too dirty to join
on. We first resolve every record to a **canonical key**: a parcel number (APN)
plus geocoded coordinates. The APN is far more stable than any address string.

### 2.3 Freshness

CoStar-style platforms refresh in batches, which is why their snapshots lag. Our
design is **event-driven**: when a deed is recorded or a permit is issued, that
event is ingested, rather than waiting for the next batch cycle.

---

## 3. Pipeline architecture

```
            Address input  (free-text string)
                   │
                   ▼
        Geocode + parcel resolution
        canonical key = APN + lat/lng
                   │
                   ▼
   ┌──────────── Parallel public-data fetch ────────────┐
   │  County records      Permits + assessor   Listings │
   │  deeds, sale, debt   capex, re-tenant      satellite│
   └────────────────────────┬───────────────────────────┘
                            ▼
                  Entity resolution
            LLC graph → beneficial owner
                            │
                            ▼
                Comp selection + NOI
             real radius, first-pass NOI
                            │
                            ▼
                  CRE-DTS risk score
            DeepSurv + LSTM + XGBoost (+ GPT-2)
```

Each stage produces structured features for the next. The entity resolution
stage is what converts a pile of per-address deeds into **borrower-level
exposure**, which is the unit credit risk actually cares about.

---

## 4. Entity resolution in detail

We treat resolution as a graph problem.

1. **Normalize** every entity name — uppercase, strip punctuation, expand
   abbreviations (`PROPS → PROPERTIES`, `CTR → CENTER`), and remove legal
   suffixes (`LLC`, `L.L.C.`, `LP`, `II`).
2. **Compare** every pair of entities on three signals:
   - **(a) fuzzy name similarity** — `token_sort_ratio` on the normalized names;
   - **(b) shared registered agent** — drawn from the recorded filing;
   - **(c) shared grantee mailing address** — where notices are sent.
3. **Link** two entities into the same beneficial owner when:

   ```
   name_sim ≥ 82
        OR
   (name_sim ≥ 55  AND  same_registered_agent  AND  same_mailing_address)
   ```

4. **Cluster** — connected components of the resulting graph are beneficial
   owners. Each cluster gets a human-readable label (the shortest normalized
   name in the cluster).

### Why the rule is deliberately conservative

A strong name match links on its own. A *moderate* name match must be
corroborated by **both** a shared agent **and** a shared mailing address before
we link. This biases toward **avoiding false merges**.

For credit risk this asymmetry matters: wrongly merging two distinct owners
understates a borrower's true concentration, which is the more dangerous error.
(Note that registered agents like CT Corporation serve tens of thousands of
unrelated companies — a shared agent alone is weak evidence, which is exactly
why the rule requires the mailing address to agree as well.)

---

## 5. Example code

A self-contained proof of concept lives in
[`entity_resolution.py`](./entity_resolution.py), running on the synthetic
[`sample_deeds.csv`](./sample_deeds.csv) (twelve deeds across five Phoenix-area
properties, with realistic name drift and chained ownership).

```bash
pip install pandas networkx rapidfuzz
python entity_resolution.py sample_deeds.csv
```

The core of the resolver:

```python
import re
import pandas as pd
import networkx as nx
from rapidfuzz import fuzz

LEGAL_SUFFIXES = [r"\bL\.?L\.?C\.?\b", r"\bL\.?P\.?\b", r"\bINC\.?\b",
                  r"\bCORP\.?\b", r"\bLTD\.?\b", r"\bCO\.?\b", r"\bII\b"]
ABBREV = {r"\bPROPS\b": "PROPERTIES", r"\bCTR\b": "CENTER",
          r"\bCAP\b": "CAPITAL", r"\bAVE\b": "AVENUE"}


def normalize_entity(name: str) -> str:
    """Collapse an entity name to a comparable canonical form."""
    s = re.sub(r"[.,]", " ", name.upper().strip())
    for pat, repl in ABBREV.items():
        s = re.sub(pat, repl, s)
    for suf in LEGAL_SUFFIXES:
        s = re.sub(suf, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_entities(df: pd.DataFrame, name_threshold: int = 82):
    """Link entity records into beneficial-owner clusters."""
    entities = pd.unique(df[["grantor", "grantee"]].values.ravel())
    norm = {e: normalize_entity(e) for e in entities}

    agent_of, mail_of = {}, {}
    for _, r in df.iterrows():
        agent_of.setdefault(r["grantee"], r["registered_agent"])
        mail_of.setdefault(r["grantee"], r["grantee_mailing_addr"])

    G = nx.Graph()
    G.add_nodes_from(entities)
    ents = list(entities)
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            sim = fuzz.token_sort_ratio(norm[a], norm[b])
            same_agent = agent_of.get(a) and agent_of.get(a) == agent_of.get(b)
            same_mail = mail_of.get(a) and mail_of.get(a) == mail_of.get(b)
            if sim >= name_threshold or (sim >= 55 and same_agent and same_mail):
                G.add_edge(a, b)

    return {e: cid for cid, comp in enumerate(nx.connected_components(G))
            for e in comp}
```

### What the run demonstrates

Two results are worth highlighting.

**Resolution survives a full rename.** `CENTRAL TOWER OWNER LLC` →
`CENTRAL TOWER OWNER, L.L.C.` → `DESERT CAP CENTRAL TOWER LLC`. The third name
fails the fuzzy-name threshold on its own, but is linked through the shared
registered agent and mailing address — the "two LLCs ago" case, resolved.

**Hidden concentration surfaces.** `CAMELBACK PLAZA PROPS 2 LLC` is the current
owner of record for *two* properties at different APNs and unrelated addresses
(450 W Camelback Rd and 2200 E Indian School Rd). At the per-address level these
look like separate owners. After resolution they aggregate to a single borrower
controlling two assets — precisely the exposure a per-address tool misses.

```
BENEFICIAL-OWNER EXPOSURE (feeds CRE-DTS borrower features)
CAMELBACK PLAZA PROPERTIES   2 property(ies)   $7,100,000 aggregate basis
CENTRAL TOWER OWNER          1 property(ies)   $21,000,000 aggregate basis
...
```

---

## 6. How this feeds the CRE-DTS model

The resolved, borrower-level output is the feature substrate for the survival
model:

| Resolution output | CRE-DTS feature |
|---|---|
| Beneficial-owner cluster | Borrower-level exposure aggregation |
| Permit / re-tenanting events | Time-varying covariates (LSTM branch) |
| Recorded debt + first-pass NOI | DSCR, LTV — survival-model inputs |
| Ownership transfer chain | Holding-period / turnover signals |

In other words, this layer answers *"who really owns this, and what else do they
own"* before the model ever estimates *"how likely is this credit to deteriorate,
and when."*

---

## 7. From proof of concept to production

The PoC is intentionally simple so the logic is legible. Scaling it raises three
engineering tasks, each mapped to a planned component of this repository.

1. **Blocking for scale.** The PoC compares every pair of entities — `O(n²)`,
   fine for twelve deeds, intractable county-wide. Production introduces
   *blocking*: partition entities by APN prefix, name token, or agent, and only
   compare within a block.

2. **Learned normalization (GPT-2 component).** The current normalizer is
   regex-based. Real deed text — legal descriptions, OCR errors, mixed-language
   entries — is far messier. The planned GPT-2 stage replaces hand-coded rules
   with learned entity-name embeddings for normalization and disambiguation.

3. **Piercing the SPE structure.** Shared-agent evidence is noisy. Truly
   resolving beneficial ownership means incorporating state Secretary-of-State
   filings and FinCEN Beneficial Ownership Information (BOI) reporting. This is
   also part of the public-interest rationale for the broader framework: better
   beneficial-ownership transparency reduces hidden CRE concentration risk for
   U.S. financial institutions.

---

## 8. Build order

The model integrates a stacking ensemble in four phases:

- **Phase 1** — C-index baseline + preprocessing (DeepSurv)
- **Phase 2** — LSTM branch (time-varying covariates)
- **Phase 3** — XGBoost branch (tabular features)
- **Phase 4** — GPT-2 branch (unstructured text, incl. entity normalization)

The data layer in this document is the input to Phase 1 and supplies the
time-varying signals consumed in Phase 2.

---
