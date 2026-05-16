import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    csv_path: str
    time_col: str
    speed_col: Optional[str]
    flow_col: Optional[str]
    occupancy_col: Optional[str]
    delimiter: str


def _parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def load_data(cfg: Config) -> pd.DataFrame:
    try:
        df = pd.read_csv(cfg.csv_path, sep=cfg.delimiter)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV topilmadi: {cfg.csv_path}")
    except Exception as e:
        raise RuntimeError(f"CSV o‘qishda xatolik: {e}")

    if cfg.time_col not in df.columns:
        raise ValueError(f"time_col='{cfg.time_col}' topilmadi. Mavjud ustunlar: {list(df.columns)}")


    df[cfg.time_col] = _parse_time(df[cfg.time_col])
    before = len(df)
    df = df.dropna(subset=[cfg.time_col]).copy()
    after = len(df)
    if after == 0:
        raise ValueError("time_col bo‘yicha barcha qiymatlar parse bo‘lmadi. Time formatini tekshiring.")

    # Soat (peak hours tahlili uchun)
    df["hour"] = df[cfg.time_col].dt.hour
    df["weekday"] = df[cfg.time_col].dt.dayofweek  # 0=Mon

    return df


def compute_peak_hours(df: pd.DataFrame, flow_col: Optional[str]) -> pd.DataFrame:
    if not flow_col or flow_col not in df.columns:
        # Har qanday holatda ham soat bo‘yicha count qaytaramiz
        peak = df.groupby("hour").size().reset_index(name="records")
        return peak.sort_values("records", ascending=False)

    g = df.groupby("hour")[flow_col].agg(["mean", "sum", "count"]).reset_index()
    g = g.rename(columns={"mean": "avg_flow", "sum": "total_flow"})
    return g.sort_values("avg_flow", ascending=False)


def traffic_summary(
    df: pd.DataFrame,
    speed_col: Optional[str],
    flow_col: Optional[str],
    occupancy_col: Optional[str],
) -> pd.DataFrame:
    rows = []

    def add_metric(name: str, col: Optional[str]):
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            rows.append(
                {
                    "metric": name,
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean(skipna=True)),
                    "median": float(s.median(skipna=True)),
                    "min": float(s.min(skipna=True)),
                    "max": float(s.max(skipna=True)),
                    "std": float(s.std(skipna=True)),
                }
            )

    add_metric("speed", speed_col)
    add_metric("flow", flow_col)
    add_metric("occupancy", occupancy_col)

    return pd.DataFrame(rows)


def anomaly_detection(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"anomaly col '{col}' topilmadi")

    x = pd.to_numeric(df[col], errors="coerce")
    # Robust z-score (Median/MAD)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if mad == 0 or np.isnan(mad):
        # fallback z-score
        mu = np.nanmean(x)
        sigma = np.nanstd(x)
        if sigma == 0 or np.isnan(sigma):
            out = pd.DataFrame({"value": [], "anomaly": []})
            return out
        z = (x - mu) / sigma
        thresh = 3.5
        mask = np.abs(z) > thresh
        out = df.loc[mask, [col, "hour", "weekday"]].copy()
        out["anomaly_score"] = z[mask]
        out = out.sort_values("anomaly_score", ascending=False)
        return out

    robust_z = 0.6745 * (x - med) / mad
    thresh = 3.5
    mask = np.abs(robust_z) > thresh

    out = df.loc[mask, [col, "hour", "weekday"]].copy()
    out["anomaly_score"] = robust_z[mask]
    out = out.sort_values("anomaly_score", ascending=False)
    return out


def export_markdown_report(
    df: pd.DataFrame,
    cfg: Config,
    speed_col: Optional[str],
    flow_col: Optional[str],
    occupancy_col: Optional[str],
    anomaly_col: Optional[str],
    out_path: str,
) -> None:
    summary = traffic_summary(df, speed_col, flow_col, occupancy_col)
    peak = compute_peak_hours(df, flow_col)

    anomaly_md = "*(anomaly detection o‘chirildi)*"
    if anomaly_col:
        anom = anomaly_detection(df, anomaly_col)
        if len(anom) == 0:
            anomaly_md = "Hech qanday anomaliya topilmadi (robust z-score bo‘yicha)."
        else:
            anomaly_md = anom.head(30).to_markdown(index=False)

    # peak table top 10
    peak_md = peak.head(10).to_markdown(index=False)

    report = []
    report.append(f"# Trafik tahlili hisoboti\n")
    report.append(f"- Fayl: `{cfg.csv_path}`\n")
    report.append(f"- Vaqt ustuni: `{cfg.time_col}`\n")
    report.append(f"- Speed: `{speed_col or '-'}'\n")
    report.append(f"- Flow: `{flow_col or '-'}'\n")
    report.append(f"- Occupancy: `{occupancy_col or '-'}'\n")
    report.append("\n## Summary\n")
    if len(summary) == 0:
        report.append("(hech qanday metrika topilmadi — ustun nomlarini tekshiring)\n")
    else:
        report.append(summary.to_markdown(index=False))
        report.append("\n")

    report.append("\n## Peak hours (top-10)\n")
    report.append(peak_md)

    report.append("\n## Anomaly detection\n")
    report.append(anomaly_md)

    report_text = "\n".join(report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)


def main(argv=None):
    p = argparse.ArgumentParser(description="Trafik tahlili: peak hours + summary + anomaly")
    p.add_argument("--csv", required=True, help="CSV fayl yo‘li")
    p.add_argument("--time-col", default="time", help="Vaqt ustuni (default: time)")
    p.add_argument("--speed-col", default="speed", help="Speed ustuni (default: speed, bo‘sh qoldirsangiz -) va qiymat topilmasa avtomatik o‘tkazib yuboriladi")
    p.add_argument("--flow-col", default="flow", help="Flow ustuni (default: flow)")
    p.add_argument("--occupancy-col", default="occupancy", help="Occupancy ustuni (default: occupancy)")
    p.add_argument("--delimiter", default=",", help="CSV ajratgich (default: ,)")
    p.add_argument("--anomaly-col", default="", help="Anomaliya aniqlash uchun ustun nomi (default: o‘chirilgan)")
    p.add_argument("--out", default="traffic_report.md", help="Hisobot chiqish fayli (default: traffic_report.md)")

    args = p.parse_args(argv)

    def norm_col(x: str) -> Optional[str]:
        if x is None:
            return None
        x = str(x).strip()
        if x == "" or x == "-":
            return None
        return x

    cfg = Config(
        csv_path=args.csv,
        time_col=args.time_col,
        speed_col=norm_col(args.speed_col),
        flow_col=norm_col(args.flow_col),
        occupancy_col=norm_col(args.occupancy_col),
        delimiter=args.delimiter,
    )

    df = load_data(cfg)

    export_markdown_report(
        df=df,
        cfg=cfg,
        speed_col=cfg.speed_col,
        flow_col=cfg.flow_col,
        occupancy_col=cfg.occupancy_col,
        anomaly_col=norm_col(args.anomaly_col),
        out_path=args.out,
    )

    print(f"OK: report yozildi -> {args.out}")


if __name__ == "__main__":
    main()

