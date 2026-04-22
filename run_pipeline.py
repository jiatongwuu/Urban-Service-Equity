from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Defaults (mirrors DB_MVP.ipynb)
# -----------------------------

# Primary data path (Google Colab + Drive). Override with --input or env MERGED_RENT_311_CSV.
DEFAULT_INPUT_CSV = os.environ.get(
    "MERGED_RENT_311_CSV",
    "/content/drive/My Drive/243 Group 2/Module 2/data/merged_rent_311.csv",
)

CLUSTER_FEATURES_DEFAULT: List[str] = [
    "median_rent",  # Economic baseline
    "median_resolution_days",  # Service efficiency
    "median_property_age",  # Infrastructure health
    "request_intensity",  # Systemic service load per unit
    "pct_311_external_request",  # Bureaucratic friction / referral rate
    "pct_street_and_sidewalk_cleaning",  # High-effort task share
]

SERVICE_COLS_DEFAULT: List[str] = [
    "total_311_requests",
    "avg_resolution_days",
    "median_resolution_days",
    "num_unique_services",
    "pct_street_and_sidewalk_cleaning",
    "pct_tree_maintenance",
    "pct_streetlights",
    "pct_rec_and_park_requests",
    "pct_graffiti",
    "pct_encampments",
    "pct_illegal_postings",
    "pct_abandoned_vehicle",
    "pct_noise_report",
]

NEED_COLS_DEFAULT: List[str] = [
    "unit_count_clean",
    "bedrooms_for_ratio",
    "sqft_avg",
    "sqft_per_resident",
    "bathrooms_per_resident",
    "property_age",
    "likely_rent_controlled",
    "monthly_rent_clean",
    "occupancy_duration_years",
]

CLUSTER_NAMES_DEFAULT: Dict[int, str] = {
    0: "Cluster A — High-Rent, Low Service Volume",
    1: "Cluster B — Dominant Residential",
    2: "Cluster C — High-Density Outlier",
    3: "Cluster D — Slow Resolution Hotspot",
}

# Heuristic archetypes (mirrors DB_MVP.ipynb Cell 10)
HEURISTICS_DEFAULT: Dict[int, dict] = {
    0: {
        "archetype": "High-Rent, Low Service Volume",
        "signal": "Newer stock + lower 311 demand — may mask under-reporting by renters",
        "needs": [
            {
                "rank": 1,
                "priority": "HIGH",
                "title": "Proactive Property Maintenance Outreach",
                "desc": "High rent + low service volume may mask deferred maintenance or tenant reluctance to report.",
                "actions": [
                    "Conduct proactive inspections in rent-controlled units",
                    "Deploy predictive maintenance alerts for aging infrastructure",
                    "Create landlord accountability portal for 311 requests",
                ],
            },
            {
                "rank": 2,
                "priority": "HIGH",
                "title": "Affordability & Displacement Prevention",
                "desc": "High rent with moderate service response. Residents may under-report fearing eviction.",
                "actions": [
                    "Expand multilingual tenant rights education and hotline",
                    "Partner with community orgs for confidential 311 submission",
                    "Track displacement-correlated service gaps longitudinally",
                ],
            },
            {
                "rank": 3,
                "priority": "MED",
                "title": "Service Volume Equity Audit",
                "desc": "Low 311 volume in high-rent areas may signal demand suppression, not satisfaction.",
                "actions": [
                    "Compare utilization vs. physical infrastructure condition",
                    "Investigate service equity for low-income renters in high-rent zones",
                    "Establish per-income-tier equity benchmarks within cluster",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Audit rent-controlled units for deferred maintenance backlog",
                "High rent + low volume is a suppression signal, not a success metric.",
            ),
            (
                "02",
                "Deploy multilingual 311 outreach in high-rent displacement corridors",
                "Non-English speakers in high-rent zones are least likely to self-report.",
            ),
            (
                "03",
                "Cross-reference 311 gaps with housing court filings",
                "Litigation spikes often precede 311 volume — early warning system.",
            ),
        ],
    },
    1: {
        "archetype": "Dominant Residential",
        "signal": "High volume, moderate affordability stress, elevated cleaning demand",
        "needs": [
            {
                "rank": 1,
                "priority": "HIGH",
                "title": "Street & Sidewalk Cleaning Scale-Up",
                "desc": "Highest cleaning request rate citywide — signals systemic under-staffing.",
                "actions": [
                    "Increase street sweeping frequency in high-density corridors",
                    "Install additional litter infrastructure at chokepoints",
                    "Pilot automated cleaning notification via 311 app",
                ],
            },
            {
                "rank": 2,
                "priority": "MED",
                "title": "Tree Canopy & Green Space Maintenance",
                "desc": "Elevated tree maintenance requests correlated with aging street trees.",
                "actions": [
                    "Prioritize tree inventory audit in blocks with highest 311 tree requests",
                    "Establish community stewardship programs for green corridors",
                    "Accelerate DPW tree trimming cycle from 10yr to 5yr target",
                ],
            },
            {
                "rank": 3,
                "priority": "HIGH",
                "title": "Housing Density Pressure Relief",
                "desc": "High occupancy ratio suggests units may be over-occupied relative to capacity.",
                "actions": [
                    "Expand secondary unit permitting in under-utilized parcels",
                    "Increase housing inspection capacity for overcrowding reports",
                    "Target density relief through inclusionary infill housing pipeline",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Surge street cleaning crews to top 20% density grid cells",
                "Cleaning requests are proportional to unmet daily maintenance burden.",
            ),
            (
                "02",
                "Fast-track tree trimming in corridors with 3+ open requests",
                "Deferred tree maintenance creates escalating liability and safety risk.",
            ),
            (
                "03",
                "Conduct housing overcrowding inspection sweep in flagged cells",
                "High density + long occupancy duration = silent overcrowding growth.",
            ),
        ],
    },
    2: {
        "archetype": "High-Density Outlier",
        "signal": "Extreme volume + severe crowding + minimal positive services — statistical anomaly",
        "needs": [
            {
                "rank": 1,
                "priority": "CRITICAL",
                "title": "Emergency High-Volume Service Infrastructure",
                "desc": "Exceptional 311 volume + critically low positive service delivery — system overwhelmed.",
                "actions": [
                    "Deploy dedicated service crew to this cell immediately",
                    "Triage 311 backlog by complaint age and recurrence frequency",
                    "Assess if this cell is a routing anomaly or genuine hotspot",
                ],
            },
            {
                "rank": 2,
                "priority": "CRITICAL",
                "title": "Crowding Investigation & Housing Audit",
                "desc": "Severe space crowding relative to building capacity. Likely undocumented multi-family.",
                "actions": [
                    "Immediate housing inspection referral",
                    "Cross-check building permits against occupancy data",
                    "Engage community health workers for vulnerable resident outreach",
                ],
            },
            {
                "rank": 3,
                "priority": "HIGH",
                "title": "Public Space Access Deficit",
                "desc": "Low positive service rate in high-density zone creates compounded livability pressure.",
                "actions": [
                    "Prioritize parklet or open space installation in adjacent blocks",
                    "Increase sidewalk maintenance inspection frequency",
                    "Establish community liaison for recurring public space issues",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Immediate multi-agency triage — 5sigma statistical outlier",
                "Volume + crowding + low services = highest composite risk in dataset.",
            ),
            (
                "02",
                "Deploy housing inspector and community health worker jointly",
                "Co-deployment reduces duplicate visits and builds resident trust.",
            ),
            (
                "03",
                "Escalate to supervisor district for emergency resource allocation",
                "Single-cell anomaly at this scale warrants political escalation.",
            ),
        ],
    },
    3: {
        "archetype": "Slow Resolution Hotspot",
        "signal": "Resolution time 5.5 sigma above city mean — institutional failure zone",
        "needs": [
            {
                "rank": 1,
                "priority": "CRITICAL",
                "title": "Emergency Resolution Time Intervention",
                "desc": "Resolution days are 5.5 standard deviations above city mean — systemic failure.",
                "actions": [
                    "Audit all open 311 tickets in this cluster older than 30 days",
                    "Assign dedicated case manager to chronic unresolved complaints",
                    "Auto-escalate any ticket open more than 14 days",
                ],
            },
            {
                "rank": 2,
                "priority": "HIGH",
                "title": "Encampment & Disorder Stabilization",
                "desc": "Elevated negative signals (encampments, graffiti, abandoned vehicles) compound slow resolution.",
                "actions": [
                    "Coordinate HSOC and DPW joint sweeps with rehousing referrals",
                    "Deploy graffiti abatement within 48h of report",
                    "Establish abandoned vehicle fast-track tow program",
                ],
            },
            {
                "rank": 3,
                "priority": "MED",
                "title": "Systemic Service Routing Reform",
                "desc": "Slow resolution often caused by inter-agency routing inefficiencies.",
                "actions": [
                    "Audit 311 routing logs for inter-agency handoff failures",
                    "Implement case ownership accountability by supervisor district",
                    "Create resolution SLA dashboard visible to department heads",
                ],
            },
        ],
        "queue": [
            (
                "01",
                "Audit all 311 tickets older than 30 days in Cluster D",
                "A 5.5sigma resolution delay is institutional failure, not backlog.",
            ),
            (
                "02",
                "Joint HSOC + DPW deployment for encampment clearance with rehousing",
                "Co-deployment with rehousing prevents cycling; clearance alone does not.",
            ),
            (
                "03",
                "Implement 14-day auto-escalation rule for all Cluster D tickets",
                "Without structural escalation triggers, chronic delays persist indefinitely.",
            ),
        ],
    },
}

# Per-grid (point-level) need/solution templates keyed by pipeline features S* / N*
POINT_FEATURE_TEMPLATES: Dict[str, Dict[str, object]] = {
    "S1": {
        "label": "Elevated 311 service load",
        "desc": "This grid is above the citywide norm on scaled 311 request volume, indicating sustained demand on local services and infrastructure.",
        "priority": lambda z: "HIGH" if z > 1.2 else "MED",
        "actions": [
            "Triage open requests by age and recurrence; add surge capacity in this grid for street and sidewalk work.",
            "Pair high-volume request blocks with a single district liaison to stop duplicate routing across agencies.",
            "Offer proactive inspections where repeat categories cluster, not only complaint-driven visits.",
        ],
    },
    "S2": {
        "label": "Slow 311 case resolution (relative to city norm)",
        "desc": "Resolution time signals are lower than a typical part of the city after scaling—cases tend to run longer to close here.",
        "priority": lambda z: "CRITICAL" if z < -1.5 else "HIGH" if z < -0.8 else "MED",
        "actions": [
            "Audit any tickets open longer than 30 days; assign an owner to older cases instead of re-routing in circles.",
            "Escalate repeat addresses to a small weekly huddle (DPW, DPH, HSOC as relevant) until backlog clears.",
            "Publish an SLA for first response in this area so residents see predictable timelines.",
        ],
    },
    "S3": {
        "label": "Low service diversity in 311 mix",
        "desc": "The mix of 311 service types here is more concentrated than a typical area—suggesting a narrower set of service channels or a routing bottleneck.",
        "priority": lambda z: "MED",
        "actions": [
            "Check whether categories are under-filed (online vs phone) and add multilingual help for filing.",
            "Route a sample of “wrong department” handoffs to root-cause the narrow mix.",
        ],
    },
    "S4_pos": {
        "label": "Underweight positive amenities & maintenance",
        "desc": "Streets, trees, rec/parks, and similar proactive requests are a smaller share than typical, which can understate unmet need.",
        "priority": lambda z: "MED",
        "actions": [
            "Schedule a corridor walk to log maintenance needs that residents may not 311 for.",
            "Prioritize one visible improvement (trees, streetlights, or park maintenance) to lift utilization trust.",
        ],
    },
    "S4_neg": {
        "label": "Disorder, safety, and encampment-related pressure",
        "desc": "A larger share of requests here is in encampment, graffiti, vehicle, and similar categories relative to the city—livability and safety are under stress.",
        "priority": lambda z: "CRITICAL" if z > 1.5 else "HIGH",
        "actions": [
            "Coordinate a joint response (street cleaning + outreach + rehousing) instead of one-off clearances.",
            "Target graffiti and abandoned-vehicle backlogs in the worst blocks on a 48–72h cycle.",
        ],
    },
    "N1": {
        "label": "Housing density and unit mix pressure",
        "desc": "The ratio of people and bedrooms to available space is high compared with typical—consistent with more intensive use of the housing stock.",
        "priority": lambda z: "HIGH" if z > 1.0 else "MED",
        "actions": [
            "Pair housing inspections with voluntary tenant interviews (community org present) to surface overcrowding safely.",
            "Map illegal conversions and subunits against permits to close the riskiest gaps first.",
        ],
    },
    "N2": {
        "label": "Space and crowding stress",
        "desc": "Crowding-related signals (space per resident) are more extreme than in most of the city.",
        "priority": lambda z: "CRITICAL" if z > 1.5 else "HIGH",
        "actions": [
            "Offer relocation assistance and legal aid where crowding is tied to unaffordability.",
            "Fast-track any secondary-unit permits that safely add capacity on underused lots nearby.",
        ],
    },
    "N3": {
        "label": "Older stock and rent-control exposure",
        "desc": "Property age and rent-control mix suggest deferred maintenance and tenant stability issues may be concentrated here.",
        "priority": lambda z: "MED",
        "actions": [
            "Run targeted health-and-safety outreach to rent-controlled stock with the worst maintenance histories.",
            "Bundle small grants for emergency repairs to prevent displacement from code violations.",
        ],
    },
    "N4": {
        "label": "Affordability and rent burden",
        "desc": "Affordability need signals are higher than a typical area—families are stretched relative to their housing options.",
        "priority": lambda z: "HIGH" if z > 0.8 else "MED",
        "actions": [
            "Connect high-burden blocks to in-language counseling on rights, benefits, and relocation payments.",
            "Prioritize affordable infill and basement legalization where the pipeline already allows it.",
        ],
    },
    "N5": {
        "label": "Tenure and turnover risk",
        "desc": "Shorter or unstable tenure patterns show up more strongly here, which can reduce political voice and maintenance investment.",
        "priority": lambda z: "MED",
        "actions": [
            "Stabilize turnover with outreach at natural lease-renewal windows.",
            "Track eviction-adjacent 311 surges as an early warning for displacement.",
        ],
    },
}


@dataclass(frozen=True)
class Config:
    k: int = 4
    random_state: int = 42
    cluster_features: Tuple[str, ...] = tuple(CLUSTER_FEATURES_DEFAULT)
    service_cols: Tuple[str, ...] = tuple(SERVICE_COLS_DEFAULT)
    need_cols: Tuple[str, ...] = tuple(NEED_COLS_DEFAULT)
    cluster_names: Dict[int, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cluster_names is None:
            object.__setattr__(self, "cluster_names", dict(CLUSTER_NAMES_DEFAULT))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: dict) -> None:
    def _default(obj: object) -> object:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_default)


def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def load_and_aggregate_to_grid(csv_path: str) -> pd.DataFrame:
    """
    Unit-level CSV -> grid-level dataframe, matching DB_MVP.ipynb Cell 3.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    _require_columns(df, ["grid_id", "lat", "lon"], context="grid aggregation")

    pct_cols = [c for c in df.columns if c.startswith("pct_")]

    agg_dict: Dict[str, str] = {
        "lat": "mean",
        "lon": "mean",
        "monthly_rent_clean": "median",
        "avg_resolution_days": "median",
        "median_resolution_days": "median",
        "total_311_requests": "sum",
        "num_unique_services": "median",
        "property_age": "median",
        "likely_rent_controlled": "mean",
        "sqft_avg": "median",
        "sqft_per_resident": "median",
        "bathrooms_per_resident": "median",
        "unit_count_clean": "sum",
        "bedrooms_for_ratio": "median",
        "occupancy_duration_years": "median",
    }

    # Only aggregate columns that exist in the input schema
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    for c in pct_cols:
        agg_dict[c] = "mean"

    grid_df = df.groupby("grid_id").agg(agg_dict).reset_index()

    # Convenience aliases for clustering feature names
    if "monthly_rent_clean" in grid_df.columns:
        grid_df["median_rent"] = grid_df["monthly_rent_clean"]
    if "property_age" in grid_df.columns:
        grid_df["median_property_age"] = grid_df["property_age"]
    if "unit_count_clean" in grid_df.columns:
        grid_df["housing_density"] = grid_df["unit_count_clean"]

    # Derived feature: 311 requests per housing unit (systemic load indicator)
    if "total_311_requests" in grid_df.columns and "housing_density" in grid_df.columns:
        denom = grid_df["housing_density"].replace(0, 1)
        grid_df["request_intensity"] = grid_df["total_311_requests"] / denom

    # Parse row/col from grid_id (format: 'row_col')
    parts = grid_df["grid_id"].astype(str).str.split("_", expand=True)
    if parts.shape[1] >= 2:
        grid_df["grid_row"] = pd.to_numeric(parts[0], errors="coerce")
        grid_df["grid_col"] = pd.to_numeric(parts[1], errors="coerce")

    return grid_df


def run_kmeans(
    grid_df: pd.DataFrame, *, cfg: Config
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Match DB_MVP.ipynb Cell 4: standardize CLUSTER_FEATURES and KMeans(K=4).
    """
    _require_columns(grid_df, list(cfg.cluster_features), context="clustering")

    df_clust = grid_df.dropna(subset=list(cfg.cluster_features)).copy()
    scaler_c = StandardScaler()
    scaled_c = scaler_c.fit_transform(df_clust[list(cfg.cluster_features)])

    kmeans = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init=10)
    df_clust["cluster"] = kmeans.fit_predict(scaled_c)

    return df_clust, kmeans, scaler_c


def equity_feature_engineering(df_clust: pd.DataFrame, *, cfg: Config) -> pd.DataFrame:
    """
    Match DB_MVP.ipynb Cell 6.
    """
    all_eq_cols = list(cfg.service_cols) + list(cfg.need_cols)
    _require_columns(df_clust, ["grid_id", "cluster", "lat", "lon"], context="equity scoring base")
    _require_columns(df_clust, all_eq_cols, context="equity scoring inputs")

    df_eq = df_clust[["grid_id", "cluster", "lat", "lon"] + all_eq_cols].dropna().copy()

    # Service Performance sub-indicators
    df_eq["S1"] = np.log1p(df_eq["total_311_requests"])
    df_eq["S2"] = (-0.5 * df_eq["avg_resolution_days"]) + (-0.5 * df_eq["median_resolution_days"])
    df_eq["S3"] = df_eq["num_unique_services"]
    df_eq["S4_pos"] = df_eq[
        [
            "pct_street_and_sidewalk_cleaning",
            "pct_tree_maintenance",
            "pct_streetlights",
            "pct_rec_and_park_requests",
        ]
    ].sum(axis=1)
    df_eq["S4_neg"] = df_eq[
        [
            "pct_encampments",
            "pct_graffiti",
            "pct_illegal_postings",
            "pct_abandoned_vehicle",
            "pct_noise_report",
        ]
    ].sum(axis=1)

    # Service Need sub-indicators
    df_eq["N1"] = (df_eq["unit_count_clean"] * df_eq["bedrooms_for_ratio"]) / df_eq["sqft_avg"].replace(
        0, np.nan
    )
    df_eq["N2"] = (1 / df_eq["sqft_per_resident"].replace(0, np.nan)) + (
        1 / df_eq["bathrooms_per_resident"].replace(0, np.nan)
    )
    df_eq["N3"] = df_eq["property_age"] + df_eq["likely_rent_controlled"]
    df_eq["N4"] = 1 / df_eq["monthly_rent_clean"].replace(0, np.nan)
    df_eq["N5"] = 1 / df_eq["occupancy_duration_years"].replace(0, np.nan)

    s_cols = ["S1", "S2", "S3", "S4_pos", "S4_neg"]
    n_cols = ["N1", "N2", "N3", "N4", "N5"]

    df_eq = df_eq[["grid_id", "cluster", "lat", "lon"] + s_cols + n_cols]
    df_eq = df_eq.replace([np.inf, -np.inf], np.nan).dropna()
    return df_eq


def compute_equity_scores(df_eq: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Match DB_MVP.ipynb Cell 7: PCA(n_components=1) per block, weighted sums,
    raw_equity ratio, log+clip+minmax -> 0-100 equity_score.
    """
    s_cols = ["S1", "S2", "S3", "S4_pos", "S4_neg"]
    n_cols = ["N1", "N2", "N3", "N4", "N5"]

    scaler_s = StandardScaler()
    scaler_n = StandardScaler()
    X_s = scaler_s.fit_transform(df_eq[s_cols])
    X_n = scaler_n.fit_transform(df_eq[n_cols])

    pca_s = PCA(n_components=1).fit(X_s)
    pca_n = PCA(n_components=1).fit(X_n)

    ws = pca_s.components_[0]
    wn = pca_n.components_[0]

    df_eq = df_eq.copy()
    df_eq["performance_score"] = (
        ws[0] * df_eq["S1"]
        + ws[1] * df_eq["S2"]
        + ws[2] * df_eq["S3"]
        + ws[3] * df_eq["S4_pos"]
        - ws[4] * df_eq["S4_neg"]
    )
    df_eq["need_score"] = (
        wn[0] * df_eq["N1"]
        + wn[1] * df_eq["N2"]
        + wn[2] * df_eq["N3"]
        + wn[3] * df_eq["N4"]
        + wn[4] * df_eq["N5"]
    )

    df_eq["raw_equity"] = df_eq["performance_score"] / (df_eq["need_score"] + 1e-6)
    log_idx = np.log1p(df_eq["raw_equity"])
    upper = np.percentile(log_idx, 99)
    clipped = np.clip(log_idx, log_idx.min(), upper)
    df_eq["equity_score"] = ((clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-6)) * 100.0

    meta = {
        "pca_weights": {
            "service_performance": dict(zip(s_cols, ws.tolist())),
            "service_need": dict(zip(n_cols, wn.tolist())),
        }
    }
    return df_eq, meta


def zscore_feature_importance(df_eq: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Root-cause style signals: cluster mean z-scores vs global mean/std.
    Returns a z-score matrix (cluster x feature) and top-3 per cluster.
    """
    features = ["S1", "S2", "S3", "S4_pos", "S4_neg", "N1", "N2", "N3", "N4", "N5"]
    mu = df_eq[features].mean()
    sd = df_eq[features].std(ddof=0).replace(0, np.nan)

    z_by_cluster = {}
    top_by_cluster = {}
    for c, sub in df_eq.groupby("cluster"):
        cm = sub[features].mean()
        z = ((cm - mu) / sd).replace([np.inf, -np.inf], np.nan)
        z_by_cluster[int(c)] = z

        top = (
            z.abs()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        top_by_cluster[int(c)] = [
            {
                "feature": feat,
                "z": float(z[feat]) if pd.notna(z[feat]) else None,
                "direction": "above_city_avg" if pd.notna(z[feat]) and z[feat] > 0 else "below_city_avg",
            }
            for feat in top
        ]

    z_df = pd.DataFrame.from_dict(z_by_cluster, orient="index")
    z_df.index.name = "cluster"
    return z_df, {"top3_features_per_cluster": top_by_cluster}


def make_cluster_summary(df_eq: pd.DataFrame, *, cfg: Config) -> pd.DataFrame:
    """
    Dashboard-friendly cluster summary: counts + equity distribution + means.
    """
    grp = df_eq.groupby("cluster")
    summary = pd.DataFrame(
        {
            "n_grids_scored": grp.size(),
            "equity_mean": grp["equity_score"].mean(),
            "equity_median": grp["equity_score"].median(),
            "equity_p10": grp["equity_score"].quantile(0.10),
            "equity_p90": grp["equity_score"].quantile(0.90),
            "performance_mean": grp["performance_score"].mean(),
            "need_mean": grp["need_score"].mean(),
        }
    ).reset_index()

    summary["cluster_name"] = summary["cluster"].map(cfg.cluster_names)
    return summary


def points_to_geojson(df: pd.DataFrame, *, id_col: str = "grid_id") -> dict:
    """
    Minimal GeoJSON FeatureCollection of Point features (no geopandas needed).
    """
    feats = []
    for row in df.itertuples(index=False):
        props = row._asdict()
        lon = props.pop("lon", None)
        lat = props.pop("lat", None)
        if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
            continue
        feats.append(
            {
                "type": "Feature",
                "id": props.get(id_col),
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {k: (None if pd.isna(v) else v) for k, v in props.items()},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _first_existing_column(df: pd.DataFrame, names: List[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _normalize_grid_id(s: object) -> str:
    t = str(s).strip()
    if t.endswith(".0") and t[:-2].isdigit():
        t = t[:-2]
    return t


def _read_optional_csv(path: str | None) -> pd.DataFrame | None:
    if not path or not os.path.isfile(path):
        return None
    return pd.read_csv(path, low_memory=False)


def _enrichment_from_rent_module2(path: str | None) -> Dict[str, dict]:
    """Per-grid aggregates from unit-level rent inventory (expects grid_id)."""
    df = _read_optional_csv(path)
    if df is None:
        return {}
    gcol = _first_existing_column(df, ["grid_id", "Grid_ID", "GRID_ID", "grid id"])
    if not gcol:
        print(f"[point advice] {path!r} has no grid_id column; skipping rent inventory enrichment.")
        return {}
    out: Dict[str, dict] = {}
    df = df.copy()
    df["_gid"] = df[gcol].map(_normalize_grid_id)
    for gid, sub in df.groupby("_gid"):
        d: dict = {}
        if "monthly_rent_clean" in sub.columns:
            med = pd.to_numeric(sub["monthly_rent_clean"], errors="coerce").median()
            if pd.notna(med):
                d["rent_median_usd"] = float(med)
        if "unit_count_clean" in sub.columns:
            d["n_inventory_units"] = int(pd.to_numeric(sub["unit_count_clean"], errors="coerce").sum())
        nhood = _first_existing_column(sub, ["analysis_neighborhood", "neighborhood", "NEIGHBORHOOD"])
        if nhood and sub[nhood].notna().any():
            d["primary_neighborhood"] = str(sub[nhood].mode().iloc[0]) if len(sub[nhood].mode()) else None
        if d:
            out[str(gid)] = d
    print(f"[point advice] Enriched {len(out)} grid(s) from rent inventory CSV.")
    return out


def _enrichment_from_grid_level_rent_311(path: str | None) -> Dict[str, dict]:
    """
    Optional grid-level file (e.g. grid_level_rent_311.csv): merge any extra descriptive fields
    that are not already in df_eq, keyed by grid_id.
    """
    df = _read_optional_csv(path)
    if df is None:
        return {}
    gcol = _first_existing_column(df, ["grid_id", "Grid_ID", "GRID_ID", "grid id"])
    if not gcol:
        print(f"[point advice] {path!r} has no grid_id column; skipping grid-level merge.")
        return {}
    skip = {gcol, "lat", "lon", "latitude", "longitude"}
    extra_cols = [c for c in df.columns if c not in skip and not str(c).startswith("Unnamed")]
    if not extra_cols:
        return {}
    out: Dict[str, dict] = {}
    df = df.copy()
    df["_gid"] = df[gcol].map(_normalize_grid_id)
    for gid, sub in df.groupby("_gid"):
        row = sub.iloc[0]
        d = {c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float, np.floating)) else str(row[c]))) for c in extra_cols[:12]}
        out[str(gid)] = {"grid_file_extras": d}
    print(f"[point advice] Merged {len(out)} row(s) from grid-level rent/311 file.")
    return out


def _enrichment_from_311_cases(path: str | None) -> Dict[str, dict]:
    """Per-grid summary from case-level 311 data (grid_id + category column)."""
    df = _read_optional_csv(path)
    if df is None:
        return {}
    gcol = _first_existing_column(df, ["grid_id", "Grid_ID", "GRID_ID", "grid id"])
    if not gcol:
        print(f"[point advice] {path!r} has no grid_id column; skipping 311 case enrichment.")
        return {}
    cat_col = _first_existing_column(
        df,
        [
            "case_type_name",
            "Case_Type",
            "Request_Category",
            "service_type",
            "Service_Type",
            "Category",
            "REQUEST_TYPE",
        ],
    )
    if not cat_col:
        print(f"[point advice] {path!r} has no recognized request category column; only counting rows per grid.")
    out: Dict[str, dict] = {}
    df = df.copy()
    df["_gid"] = df[gcol].map(_normalize_grid_id)
    for gid, sub in df.groupby("_gid"):
        d: dict = {"n_311_cases": int(len(sub))}
        if cat_col:
            modes = sub[cat_col].dropna().astype(str).value_counts()
            if len(modes):
                d["top_311_type"] = str(modes.index[0])
                d["top_311_share"] = float(modes.iloc[0] / max(len(sub), 1))
        if d:
            out[str(gid)] = d
    print(f"[point advice] Enriched {len(out)} grid(s) from 311 case file.")
    return out


def _merge_enrichments(*parts: Dict[str, dict]) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for p in parts:
        for k, v in p.items():
            if k not in merged:
                merged[k] = {}
            merged[k] = {**merged[k], **v}
    return merged


def _enrichment_text_blob(gid: str, e: dict) -> str:
    if not e:
        return ""
    parts: List[str] = []
    if e.get("rent_median_usd") is not None:
        n = e.get("n_inventory_units", "")
        parts.append(
            f"Housing inventory in this file: median rent about ${e['rent_median_usd']:,.0f}/mo"
            + (f" across {n} unit rows." if n != "" else ".")
        )
    if e.get("top_311_type"):
        parts.append(
            f"Most common 311 type in the case file: {e['top_311_type']}"
            + (f" ({e.get('top_311_share', 0) * 100:.0f}% of local cases)." if e.get("top_311_share") is not None else ".")
        )
    if e.get("n_311_cases") and not e.get("top_311_type"):
        parts.append(f"311 case file: {e['n_311_cases']:,} row(s) tagged to this grid.")
    if e.get("primary_neighborhood"):
        parts.append(f"Neighborhood (inventory): {e['primary_neighborhood']}.")
    return " ".join(parts)


def _problem_score(feat: str, z: float) -> float:
    if feat == "S2":
        return max(0.0, float(-z))
    if feat in ("S1", "S4_neg", "N1", "N2", "N3", "N4", "N5"):
        return max(0.0, float(z))
    if feat == "S3":
        return max(0.0, float(-z)) * 0.5
    if feat == "S4_pos":
        return max(0.0, float(-z)) * 0.4
    return 0.0


def build_point_level_advice(
    df_eq: pd.DataFrame,
    enrichment: Dict[str, dict],
    *,
    features: List[str] | None = None,
) -> dict:
    """
    For each grid_id, top feature-level drivers vs city (z-scores) + need cards + 3 action bullets.
    """
    feats = features or ["S1", "S2", "S3", "S4_pos", "S4_neg", "N1", "N2", "N3", "N4", "N5"]
    mu = df_eq[feats].mean()
    sd = df_eq[feats].std(ddof=0).replace(0, np.nan)

    by_grid: dict = {}
    for row in df_eq.itertuples(index=False):
        d = row._asdict()
        gid = str(d.get("grid_id", ""))
        e = enrichment.get(gid, {})

        zmap: Dict[str, float] = {}
        for f in feats:
            m, s = mu[f], sd[f]
            if pd.isna(s) or s == 0:
                zmap[f] = 0.0
            else:
                zmap[f] = float((d[f] - m) / s)

        scored = sorted(
            ((f, _problem_score(f, zmap[f]), zmap[f]) for f in feats),
            key=lambda x: -x[1],
        )
        # keep drivers with any meaningful stress
        top = [(f, s, zv) for f, s, zv in scored if s > 0.15][:3]
        if not top:
            # fall back: strongest |z| so the UI still has copy
            top = sorted(((f, abs(zmap[f]), zmap[f]) for f in feats), key=lambda x: -x[1])[:2]
            top = [(f, s, zv) for f, s, zv in top]

        needs: List[dict] = []
        action_pool: List[str] = []
        for rank, (feat, _sc, zv) in enumerate(top, start=1):
            tpl = POINT_FEATURE_TEMPLATES.get(feat)
            if not tpl:
                continue
            pr_fn = tpl["priority"]
            priority = pr_fn(zv) if callable(pr_fn) else str(pr_fn)
            needs.append(
                {
                    "rank": rank,
                    "feature": feat,
                    "z": round(zv, 3),
                    "priority": str(priority),
                    "title": str(tpl["label"]),
                    "desc": str(tpl["desc"]),
                }
            )
            for a in tpl.get("actions", [])[:4]:
                if a not in action_pool:
                    action_pool.append(a)
                if len(action_pool) >= 8:
                    break
            if len(needs) >= 3:
                break

        sol = action_pool[:3]
        if len(sol) < 3:
            sol = (sol + ["Tie public messaging to visible fixes on this block (trees, lights, or sidewalk) so 311 is trusted."])[:3]

        extra = _enrichment_text_blob(gid, e)
        if extra:
            needs.insert(
                0,
                {
                    "rank": 0,
                    "feature": "file_context",
                    "z": 0.0,
                    "priority": "INFO",
                    "title": "Local data files for this grid",
                    "desc": extra,
                },
            )

        by_grid[gid] = {
            "needs": needs,
            "solutions": sol,
        }
        if e:
            by_grid[gid]["enrichment"] = e

    return by_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering + equity scoring pipeline and export dashboard artifacts.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        help=(
            "Path to merged rent + 311 CSV (unit-level, same schema as DB_MVP.ipynb). "
            f"Default: Colab Drive path, or set MERGED_RENT_311_CSV. Default value: {DEFAULT_INPUT_CSV!r}"
        ),
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory to write outputs.")
    parser.add_argument("--k", type=int, default=4, help="KMeans clusters (default 4).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--write-geojson", action="store_true", help="Also write GeoJSON point layers.")
    parser.add_argument(
        "--rent-module2",
        default=os.path.join("data", "rent_dataset_module2.csv"),
        help="Optional housing inventory CSV (with grid_id) to enrich per-grid need/solution text.",
    )
    parser.add_argument(
        "--grid-rent-311",
        default=os.path.join("data", "grid_level_rent_311.csv"),
        help="Optional grid-level rent+311 CSV merged on grid_id (extra fields merged into point advice).",
    )
    parser.add_argument(
        "--311-data",
        default=os.path.join("data", "311_data.csv"),
        dest="cases_311",
        help="Optional 311 case-level CSV (with grid_id) for category + volume enrichment. Default: data/311_data.csv",
    )
    args = parser.parse_args()

    cfg = Config(k=args.k, random_state=args.random_state)
    _ensure_dir(args.output_dir)

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input CSV not found: {input_path!r}\n"
            "  • On Colab: mount Drive and ensure the file exists at the default path, or pass --input.\n"
            "  • Locally: pass your file explicitly, e.g. --input ./merged_rent_311.csv\n"
            "  • Or set environment variable MERGED_RENT_311_CSV to the full path."
        )

    grid_df = load_and_aggregate_to_grid(input_path)
    df_clust, kmeans, scaler_c = run_kmeans(grid_df, cfg=cfg)

    df_eq_base = equity_feature_engineering(df_clust, cfg=cfg)
    df_eq, meta = compute_equity_scores(df_eq_base)

    z_df, z_meta = zscore_feature_importance(df_eq)
    cluster_summary = make_cluster_summary(df_eq, cfg=cfg)

    e_rent = _enrichment_from_rent_module2(args.rent_module2)
    e_gr = _enrichment_from_grid_level_rent_311(args.grid_rent_311)
    e_311 = _enrichment_from_311_cases(args.cases_311)
    enrichment = _merge_enrichments(e_rent, e_gr, e_311)
    point_by_grid = build_point_level_advice(df_eq, enrichment)

    # Join top-3 features onto scored grids (for map tooltips)
    top_map = z_meta["top3_features_per_cluster"]
    df_eq["top3_features"] = df_eq["cluster"].astype(int).map(
        lambda c: ", ".join([d["feature"] for d in top_map.get(int(c), [])])
    )

    # Exports
    grid_results_path = os.path.join(args.output_dir, "grid_results.csv")
    cluster_summary_path = os.path.join(args.output_dir, "cluster_summary.csv")
    zscores_path = os.path.join(args.output_dir, "cluster_feature_zscores.csv")
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    point_advice_path = os.path.join(args.output_dir, "grid_point_advice.json")

    df_eq.to_csv(grid_results_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)
    z_df.to_csv(zscores_path)

    _write_json(
        point_advice_path,
        {
            "version": 1,
            "ingestion": {
                "rent_module2_csv": args.rent_module2,
                "grids_with_rent_file_enrichment": len(e_rent),
                "grid_rent_311_csv": args.grid_rent_311,
                "grids_with_grid_file_row": len(e_gr),
                "cases_311_csv": args.cases_311,
                "grids_with_case_stats": len(e_311),
            },
            "by_grid": point_by_grid,
        },
    )

    _write_json(
        metadata_path,
        {
            "config": {
                "k": cfg.k,
                "random_state": cfg.random_state,
                "cluster_features": list(cfg.cluster_features),
                "service_cols": list(cfg.service_cols),
                "need_cols": list(cfg.need_cols),
                "cluster_names": cfg.cluster_names,
            },
            "pca_weights": meta["pca_weights"],
            "top3_features_per_cluster": z_meta["top3_features_per_cluster"],
            "heuristics": HEURISTICS_DEFAULT,
            "artifacts": {
                "grid_results_csv": os.path.basename(grid_results_path),
                "cluster_summary_csv": os.path.basename(cluster_summary_path),
                "cluster_feature_zscores_csv": os.path.basename(zscores_path),
                "grid_point_advice_json": os.path.basename(point_advice_path),
            },
        },
    )

    if args.write_geojson:
        geo_cols = ["grid_id", "lat", "lon", "cluster", "equity_score", "performance_score", "need_score", "top3_features"]
        geo_df = df_eq[geo_cols].copy()
        geojson = points_to_geojson(geo_df)
        _write_json(os.path.join(args.output_dir, "grid_points.geojson"), geojson)

    # Lightweight console summary (so you can sanity-check large runs)
    print(f"Input: {input_path}")
    print(f"Aggregated grids: {len(grid_df):,}")
    print(f"Clustered grids (complete features): {len(df_clust):,}")
    print(f"Equity-scored grids: {len(df_eq):,}")
    print(f"Wrote: {grid_results_path}")
    print(f"Wrote: {cluster_summary_path}")
    print(f"Wrote: {zscores_path}")
    print(f"Wrote: {metadata_path}")
    print(f"Wrote: {point_advice_path} ({len(point_by_grid):,} grids)")


if __name__ == "__main__":
    main()

