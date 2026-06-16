"""
CRE Entity Resolution — Proof of Concept
=========================================
Problem: county deed records list owners as LLCs/SPEs. The same beneficial
owner shows up under many entity names across time ("two LLCs ago"). To get a
current, borrower-level view of CRE exposure, we must resolve fragmented entity
records into beneficial-owner clusters.

Pipeline:
  1. Normalize entity names (strip suffixes, punctuation, abbreviations)
  2. Build per-APN ownership transfer chains (grantor -> grantee over time)
  3. Resolve entities into clusters using three signals:
       (a) fuzzy name similarity
       (b) shared registered agent
       (c) shared grantee mailing address
  4. Identify CURRENT owner of record per APN (latest grant on chain)
  5. Aggregate exposure to the resolved beneficial owner

This is the data/entity layer that feeds borrower-level features into the
CRE-DTS survival model.
"""

import re
import sys
from collections import defaultdict

import pandas as pd
import networkx as nx
from rapidfuzz import fuzz

# ----------------------------------------------------------------------------
# 1. ENTITY NAME NORMALIZATION
# ----------------------------------------------------------------------------

LEGAL_SUFFIXES = [
    r"\bL\.?L\.?C\.?\b", r"\bL\.?P\.?\b", r"\bINC\.?\b", r"\bCORP\.?\b",
    r"\bLTD\.?\b", r"\bCO\.?\b", r"\bII\b", r"\b2\b", r"\bL L C\b",
]

ABBREV = {
    r"\bPROPS\b": "PROPERTIES",
    r"\bCTR\b": "CENTER",
    r"\bCAP\b": "CAPITAL",
    r"\bMIXED-USE\b": "MIXED USE",
    r"\bAVE\b": "AVENUE",
}


def normalize_entity(name: str) -> str:
    """Collapse an entity name to a comparable canonical form."""
    s = name.upper().strip()
    s = re.sub(r"[.,]", " ", s)
    for pat, repl in ABBREV.items():
        s = re.sub(pat, repl, s)
    for suf in LEGAL_SUFFIXES:
        s = re.sub(suf, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------
# 2 + 3. ENTITY RESOLUTION
# ----------------------------------------------------------------------------

NAME_THRESHOLD = 82  # token_sort_ratio above this => likely same entity


def resolve_entities(df: pd.DataFrame) -> dict:
    """
    Build a graph where nodes are raw entity names and edges link entities
    judged to be the same beneficial owner. Connected components = clusters.
    """
    # gather every distinct raw entity appearing as grantor or grantee
    entities = pd.unique(df[["grantor", "grantee"]].values.ravel())
    norm = {e: normalize_entity(e) for e in entities}

    # lookups for the corroborating signals (taken from grantee rows)
    agent_of = {}
    mail_of = {}
    for _, r in df.iterrows():
        agent_of.setdefault(r["grantee"], r["registered_agent"])
        mail_of.setdefault(r["grantee"], r["grantee_mailing_addr"])

    G = nx.Graph()
    G.add_nodes_from(entities)

    ents = list(entities)
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            name_sim = fuzz.token_sort_ratio(norm[a], norm[b])
            same_agent = (
                agent_of.get(a) is not None
                and agent_of.get(a) == agent_of.get(b)
            )
            same_mail = (
                mail_of.get(a) is not None
                and mail_of.get(a) == mail_of.get(b)
            )

            # Decision rule: strong name match alone, OR a moderate name match
            # corroborated by a shared agent AND shared mailing address.
            link = name_sim >= NAME_THRESHOLD or (
                name_sim >= 55 and same_agent and same_mail
            )
            if link:
                G.add_edge(a, b, name_sim=name_sim,
                           agent=same_agent, mail=same_mail)

    clusters = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for e in comp:
            clusters[e] = cid
    return clusters, G


# ----------------------------------------------------------------------------
# 4. CURRENT OWNER OF RECORD per APN
# ----------------------------------------------------------------------------

GRANTING_DOCS = {"GRANT_DEED", "CORRECTION_DEED", "QUITCLAIM"}


def current_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Latest granting deed per APN determines today's owner of record."""
    g = df[df["doc_type"].isin(GRANTING_DOCS)].copy()
    g["record_date"] = pd.to_datetime(g["record_date"])
    idx = g.groupby("apn")["record_date"].idxmax()
    latest = g.loc[idx, ["apn", "property_address", "grantee",
                         "record_date", "sale_amount"]]
    return latest.rename(columns={"grantee": "current_owner_raw",
                                  "record_date": "as_of"})


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------

def main(path: str):
    df = pd.read_csv(path)

    clusters, G = resolve_entities(df)
    n_raw = len(clusters)
    n_clusters = len(set(clusters.values()))

    # human-readable cluster labels = shortest normalized name in the cluster
    members = defaultdict(list)
    for ent, cid in clusters.items():
        members[cid].append(ent)
    label = {cid: min((normalize_entity(e) for e in ents), key=len)
             for cid, ents in members.items()}

    print("=" * 70)
    print("ENTITY RESOLUTION RESULT")
    print("=" * 70)
    print(f"Raw distinct entities : {n_raw}")
    print(f"Resolved owners       : {n_clusters}")
    print(f"Collapse ratio        : {n_raw/n_clusters:.2f}x\n")

    for cid, ents in sorted(members.items()):
        if len(ents) > 1:
            print(f"[Owner {cid}]  ->  {label[cid]}")
            for e in sorted(ents):
                print(f"     - {e}")
            print()

    # current owner per property, mapped to resolved owner
    owners = current_owner(df)
    owners["resolved_owner_id"] = owners["current_owner_raw"].map(clusters)
    owners["resolved_owner"] = owners["resolved_owner_id"].map(label)

    print("=" * 70)
    print("CURRENT OWNER OF RECORD (per property)")
    print("=" * 70)
    for _, r in owners.iterrows():
        print(f"APN {r['apn']}  {r['property_address']}")
        print(f"    owner of record : {r['current_owner_raw']}")
        print(f"    resolved owner  : {r['resolved_owner']}  "
              f"(cluster {r['resolved_owner_id']})")
        print(f"    last sale       : ${r['sale_amount']:,.0f} "
              f"on {r['as_of'].date()}")
        print()

    # borrower-level exposure aggregation -> feeds CRE-DTS
    print("=" * 70)
    print("BENEFICIAL-OWNER EXPOSURE (feeds CRE-DTS borrower features)")
    print("=" * 70)
    expo = (owners.groupby("resolved_owner")
            .agg(properties=("apn", "nunique"),
                 last_basis=("sale_amount", "sum"))
            .sort_values("last_basis", ascending=False))
    for owner, row in expo.iterrows():
        print(f"{owner:<28} {int(row['properties'])} property(ies)   "
              f"${row['last_basis']:,.0f} aggregate basis")

    # surface the hidden cross-property link explicitly
    print()
    multi = expo[expo["properties"] > 1]
    if len(multi):
        print(">> Hidden concentration found: a single resolved owner controls")
        print("   multiple properties across different APNs / addresses that")
        print("   look unrelated at the deed level. This is exactly the")
        print("   borrower-level exposure a per-address tool would miss.")

    return clusters, owners, expo


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_deeds.csv"
    main(path)
