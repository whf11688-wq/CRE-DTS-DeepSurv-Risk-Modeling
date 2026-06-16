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

*This data layer is part of an AI-enabled, forward-looking CRE credit-risk
assessment framework for U.S. financial institutions.*
