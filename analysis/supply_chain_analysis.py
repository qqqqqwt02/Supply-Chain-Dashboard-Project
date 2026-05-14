"""
Supply chain CSV analysis.

Data: ``data/supply_chain_data.csv`` (same file as the original dashboard analysis).

Run from repo root:
    python analysis/supply_chain_analysis.py

Requirements: Python 3.7+ (3.9+ recommended). See requirements.txt.
"""

from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Tuple

# Writable MPL config (CI / sandbox / some macOS Python builds)
_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT_EARLY / ".mplconfig"))

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import pandas as pd

ROOT = _ROOT_EARLY
DATA_PATH = ROOT / "data" / "supply_chain_data.csv"
OUTPUT_DIR = ROOT / "output"
BA_CASE_NARRATIVE_TXT = OUTPUT_DIR / "ba_case_narrative.txt"

STOCK_ALERT = 20
DEFECT_ALERT_PCT = 3.0


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    rename_map = {
        "Lead times": "lead_times_ops",
        "Lead time": "lead_time_supplier_days",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def implied_revenue(df: pd.DataFrame) -> pd.Series:
    return df["Price"] * df["Number of products sold"]


def _fmt_money(x: float) -> str:
    return "{:,.0f}".format(x)


def _fmt_pct(x: float, digits: int = 1) -> str:
    return "{:.{}f}%".format(x, digits)


# --- Case 1: Supplier & quality -------------------------------------------------


def build_case_supplier_quality(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """BA case: supplier scorecard — QC outcomes, defect concentration, revenue exposure."""
    fail = df["Inspection results"] == "Fail"
    df_m = df.assign(_fail=fail, _pending=df["Inspection results"] == "Pending")
    grp = df_m.groupby("Supplier name")
    by_supplier = grp.agg(
        skus=("SKU", "count"),
        fail_rate=("_fail", "mean"),
        pending_rate=("_pending", "mean"),
        avg_defect=("Defect rates", "mean"),
        revenue=("Revenue generated", "sum"),
    ).reset_index()
    rev_fail = (
        df_m.loc[df_m["_fail"], ["Supplier name", "Revenue generated"]]
        .groupby("Supplier name", as_index=False)["Revenue generated"]
        .sum()
        .rename(columns={"Revenue generated": "revenue_at_fail"})
    )
    by_supplier = by_supplier.merge(rev_fail, on="Supplier name", how="left")
    by_supplier["revenue_at_fail"] = by_supplier["revenue_at_fail"].fillna(0)
    by_supplier["fail_rate"] *= 100
    by_supplier["pending_rate"] *= 100
    by_supplier = by_supplier.sort_values("fail_rate", ascending=False)
    worst = by_supplier.iloc[0]
    portfolio_fail = fail.mean() * 100
    high_defect = (df["Defect rates"] > DEFECT_ALERT_PCT).sum()

    narrative = (
        "\n[Case 1] Supplier quality and revenue exposure (procurement / quality)\n"
        "— Business question: With limited audit capacity, which suppliers should we prioritize, "
        "and how do we quantify revenue tied to poor quality outcomes?\n"
        "— Definition: By Supplier name, aggregate SKU count, Fail/Pending share, mean Defect rates, "
        "total Revenue generated, and revenue on SKUs that Fail inspection.\n"
        "— Findings: Portfolio Fail rate is about {pf}; SKUs with defect rate > {dthr}%: {hd}. "
        "Highest Fail-rate supplier: {ws} (Fail ~{wfr}, Pending ~{wpr}; revenue on Fail SKUs ~{wrev}).\n"
        "— Recommendations: (1) Joint quality-improvement plan with {ws} and a Fail-rate target; "
        "(2) root-cause analysis on high-Fail, high-revenue SKUs (incoming material / process / spec); "
        "(3) contract quality KPIs and second-source evaluation to reduce disruption and recall risk.\n"
    ).format(
        pf=_fmt_pct(portfolio_fail),
        dthr=int(DEFECT_ALERT_PCT),
        hd=int(high_defect),
        ws=worst["Supplier name"],
        wfr=_fmt_pct(float(worst["fail_rate"])),
        wpr=_fmt_pct(float(worst["pending_rate"])),
        wrev=_fmt_money(float(worst["revenue_at_fail"])),
    )
    metrics = {
        "portfolio_fail_pct": portfolio_fail,
        "worst_supplier": worst["Supplier name"],
        "supplier_table": by_supplier,
    }
    return narrative, metrics


# --- Case 2: Inventory & service ------------------------------------------------


def build_case_inventory_service(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """BA case: stock vs availability — understock and low on-shelf risk."""
    low_stock = df["Stock levels"] < STOCK_ALERT
    low_avail = df["Availability"] < 50
    risk = low_stock & low_avail
    rev_at_risk = df.loc[risk, "Revenue generated"].sum()
    n_risk = int(risk.sum())
    avg_avail_low = df.loc[low_stock, "Availability"].mean()
    avg_avail_ok = df.loc[~low_stock, "Availability"].mean()

    top_risk = (
        df.loc[risk, ["SKU", "Product type", "Stock levels", "Availability", "Revenue generated"]]
        .sort_values("Revenue generated", ascending=False)
        .head(5)
    )

    narrative = (
        "\n[Case 2] Inventory and service level (operations / planning)\n"
        "— Business question: Which SKUs face both low on-hand stock and low sell-through / availability, "
        "putting revenue and customer experience at risk?\n"
        "— Definition: Stock levels < {sa} = low stock; Availability < 50% = low availability; "
        "both = dual-low risk SKU.\n"
        "— Findings: {nr} dual-low SKUs with combined revenue ~{rev}; "
        "mean Availability for low-stock SKUs ~{a1}, for others ~{a2} (context for stock vs availability).\n"
        "— Recommendations: (1) safety stock and replenishment cadence for dual-low, high-revenue SKUs; "
        "(2) align with demand planning on promo and seasonality; "
        "(3) SKU-level dashboard on Stock x Availability quadrants.\n"
    ).format(
        sa=STOCK_ALERT,
        nr=n_risk,
        rev=_fmt_money(float(rev_at_risk)),
        a1=_fmt_pct(float(avg_avail_low) if pd.notna(avg_avail_low) else 0.0),
        a2=_fmt_pct(float(avg_avail_ok) if pd.notna(avg_avail_ok) else 0.0),
    )
    return narrative, {"risk_skus": top_risk, "dual_low_count": n_risk}


# --- Case 3: Logistics & cost ----------------------------------------------------


def build_case_logistics_spend(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """BA case: transportation mode — unit economics and savings hypothesis."""
    df2 = df.assign(combined=df["Costs"] + df["Shipping costs"])
    mode_avg = df2.groupby("Transportation modes")["combined"].mean()
    hi_name = mode_avg.idxmax()
    lo_name = mode_avg.idxmin()
    hic = float(mode_avg.max())
    loc = float(mode_avg.min())
    gap = hic - loc
    g2 = mode_avg.reset_index().rename(columns={"combined": "avg_combined"})

    narrative = (
        "\n[Case 3] Transportation mode and total logistics cost (cost optimization)\n"
        "— Business question: How do combined route/logistics cost plus shipping differ by Transportation modes, "
        "and is there a testable savings hypothesis?\n"
        "— Definition: For each row, combined = Costs + Shipping costs; mean combined by transportation mode.\n"
        "— Findings: Highest mean combined cost: {hi} (~{hic} per row); lowest: {lo} (~{loc}); "
        "spread ~{gap} (same currency units as the dataset).\n"
        "— Recommendations: (1) segment by volume, lead time, and value to find lanes to shift to lower-cost modes; "
        "(2) tie to quality and OTIF SLAs so cost cuts do not erode service; "
        "(3) decision tree for mode selection and pilot lanes to validate savings.\n"
    ).format(
        hi=hi_name,
        hic=_fmt_money(hic),
        lo=lo_name,
        loc=_fmt_money(loc),
        gap=_fmt_money(gap),
    )
    return narrative, {"mode_costs": g2}


# --- Optional: data trust one-liner (not a full fourth case) --------------------


def data_quality_appendix_text(df: pd.DataFrame) -> str:
    implied = implied_revenue(df)
    rev = df["Revenue generated"]
    denom = rev.replace(0, float("nan"))
    mismatch = ((rev - implied).abs() / denom).median() * 100
    return (
        "Data trust: median |Revenue generated - Price x units| / Revenue is about {}. ".format(_fmt_pct(float(mismatch)))
    )


# --- Charts (aligned with cases) -------------------------------------------------


def plot_inspection_by_supplier(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["Supplier name"], df["Inspection results"], normalize="index") * 100
    for col in ["Pass", "Pending", "Fail"]:
        if col not in ct.columns:
            ct[col] = 0.0
    ct = ct[["Pass", "Pending", "Fail"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="barh", stacked=True, ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_xlabel("Share of SKUs (%)")
    ax.set_title("Case 1 — Quality mix by supplier")
    ax.legend(title="Inspection", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stock_availability(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors_pt = {"haircare": "#3498db", "skincare": "#9b59b6", "cosmetics": "#1abc9c"}
    for ptype in df["Product type"].unique():
        sub = df[df["Product type"] == ptype]
        ax.scatter(
            sub["Stock levels"],
            sub["Availability"],
            c=colors_pt.get(ptype, "#7f8c8d"),
            label=ptype,
            alpha=0.85,
        )
    ax.axvline(STOCK_ALERT, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Stock levels")
    ax.set_ylabel("Availability")
    ax.set_title("Case 2 — Stock vs availability (dashed = alert)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_logistics_combined(df: pd.DataFrame, out: Path) -> None:
    df2 = df.assign(combined=df["Costs"] + df["Shipping costs"])
    g = df2.groupby("Transportation modes", as_index=False)["combined"].mean().sort_values("combined", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g["Transportation modes"], g["combined"], color="#34495e")
    ax.set_ylabel("Avg (Costs + Shipping)")
    ax.set_title("Case 3 — Combined logistics cost by mode")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_risk_table(df: pd.DataFrame, out: Path) -> None:
    implied = implied_revenue(df)
    rev = df["Revenue generated"]
    denom = rev.replace(0, float("nan"))
    df_out = df.assign(
        revenue_implied=implied,
        revenue_variance_pct=(rev - implied).abs() / denom * 100,
        stock_risk_flag=df["Stock levels"] < STOCK_ALERT,
        defect_risk_flag=df["Defect rates"] > DEFECT_ALERT_PCT,
        qc_fail=df["Inspection results"] == "Fail",
        dual_low_risk=(df["Stock levels"] < STOCK_ALERT) & (df["Availability"] < 50),
        combined_logistics_cost=df["Costs"] + df["Shipping costs"],
    )
    cols = [
        "SKU",
        "Product type",
        "Supplier name",
        "Location",
        "Revenue generated",
        "revenue_implied",
        "revenue_variance_pct",
        "Stock levels",
        "Availability",
        "stock_risk_flag",
        "dual_low_risk",
        "Defect rates",
        "defect_risk_flag",
        "Inspection results",
        "qc_fail",
        "Transportation modes",
        "Costs",
        "Shipping costs",
        "combined_logistics_cost",
    ]
    present = [c for c in cols if c in df_out.columns]
    df_out[present].sort_values(
        ["qc_fail", "dual_low_risk", "defect_risk_flag", "revenue_variance_pct"],
        ascending=[False, False, False, False],
    ).to_csv(str(out), index=False)


def build_ba_cases_report_text(df: pd.DataFrame) -> str:
    """Same narrative as printed to stdout, for saving to a .txt file."""
    buf = StringIO()

    def w(*args, **kwargs):
        print(*args, file=buf, **kwargs)

    w("\n" + "=" * 72)
    w("Business analyst project demo — three cases from {} (ONLINE DATASET)".format(DATA_PATH.name))
    w("=" * 72)

    t1, _ = build_case_supplier_quality(df)
    w(t1.rstrip("\n"))
    t2, m2 = build_case_inventory_service(df)
    w(t2.rstrip("\n"))
    if not m2["risk_skus"].empty:
        w("   Top 5 dual-low SKUs by revenue:")
        w(m2["risk_skus"].to_string(index=False))
    t3, _ = build_case_logistics_spend(df)
    w(t3.rstrip("\n"))

    w(data_quality_appendix_text(df))
    w("\n" + "=" * 72)
    return buf.getvalue()


def print_ba_cases(df: pd.DataFrame) -> str:
    text = build_ba_cases_report_text(df)
    print(text, end="")
    return text


def main() -> int:
    if not DATA_PATH.exists():
        print("Missing data file: {}".format(DATA_PATH), file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True})

    df = load_data()
    narrative_text = print_ba_cases(df)
    BA_CASE_NARRATIVE_TXT.write_text(narrative_text, encoding="utf-8")

    plot_inspection_by_supplier(df, OUTPUT_DIR / "ba_case1_quality_by_supplier.png")
    plot_stock_availability(df, OUTPUT_DIR / "ba_case2_stock_vs_availability.png")
    plot_logistics_combined(df, OUTPUT_DIR / "ba_case3_combined_cost_by_mode.png")
    export_risk_table(df, OUTPUT_DIR / "ba_sku_analyst_flags.csv")

    print("Charts, detail table, and narrative text written to: {}".format(OUTPUT_DIR))
    print("Read the cases in: {}".format(BA_CASE_NARRATIVE_TXT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
