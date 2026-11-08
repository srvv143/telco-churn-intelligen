"""Lightweight data-contract validation (schema, domain levels, ranges, consistency)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import CATEGORY_LEVELS, ID_COLUMN, INTERNET_ADDONS, NUMERIC_RANGES, TARGET


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("Data validation failed:\n  - " + "\n  - ".join(self.errors))

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def validate_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
    required_columns: list[str] | None = None,
) -> ValidationReport:
    """Validate a *cleaned* frame. Errors block training/scoring; warnings are informational."""
    report = ValidationReport(n_rows=len(df))
    required = (
        list(required_columns)
        if required_columns is not None
        else list(CATEGORY_LEVELS) + list(NUMERIC_RANGES)
    )
    if require_target:
        required.append(TARGET)

    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"missing required columns: {missing}")

    for col, levels in CATEGORY_LEVELS.items():
        if col not in df.columns:
            continue
        values = df[col].dropna()
        bad = sorted(set(values.astype(str)) - set(levels))
        if bad:
            report.errors.append(f"{col}: unexpected levels {bad[:5]} (allowed {list(levels)})")
        if df[col].isna().any():
            report.warnings.append(f"{col}: {int(df[col].isna().sum())} missing values (imputed)")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out_of_range = values.notna() & ((values < lo) | (values > hi))
        if out_of_range.any():
            report.errors.append(f"{col}: {int(out_of_range.sum())} values outside [{lo}, {hi}]")
        n_missing = int(values.isna().sum())
        if n_missing:
            report.warnings.append(f"{col}: {n_missing} missing values (imputed in pipeline)")

    if ID_COLUMN in df.columns and df[ID_COLUMN].duplicated().any():
        report.errors.append(f"{ID_COLUMN}: {int(df[ID_COLUMN].duplicated().sum())} duplicated ids")

    if require_target and TARGET in df.columns:
        bad_target = sorted(set(df[TARGET].dropna().astype(str)) - {"Yes", "No", "0", "1"})
        if bad_target:
            report.errors.append(f"{TARGET}: unexpected values {bad_target}")

    # Cross-field consistency of the service hierarchy.
    if {"InternetService", *INTERNET_ADDONS} <= set(df.columns):
        no_net = df["InternetService"] == "No"
        for addon in INTERNET_ADDONS:
            bad = no_net & (df[addon] != "No internet service")
            if bad.any():
                report.warnings.append(
                    f"{addon}: {int(bad.sum())} rows inconsistent with InternetService='No'"
                )
    if {"PhoneService", "MultipleLines"} <= set(df.columns):
        bad = (df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")
        if bad.any():
            report.warnings.append(
                f"MultipleLines: {int(bad.sum())} rows inconsistent with PhoneService='No'"
            )
    return report
