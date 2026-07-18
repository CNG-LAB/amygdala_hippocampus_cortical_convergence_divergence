#!/usr/bin/env python3
"""Summarize age and sex for the configured subject cohort."""

import os

import pandas as pd


def main() -> None:
    subject_list = os.environ["HIPPAMYG_SUBJECT_LIST"]
    age_csv = os.environ["HCP_RESTRICTED_CSV"]
    gender_csv = os.environ["HCP_UNRESTRICTED_CSV"]

    with open(subject_list, "r") as f:
        raw_ids = [line.strip() for line in f if line.strip()]
    subject_ids = pd.to_numeric(raw_ids, errors="coerce")
    subject_ids = set(subject_ids[~pd.isna(subject_ids)].astype(int))

    age = pd.read_csv(age_csv, usecols=["Subject", "Age_in_Yrs"])
    sex = pd.read_csv(gender_csv, usecols=["Subject", "Gender"])
    age["Subject"] = pd.to_numeric(age["Subject"], errors="coerce")
    sex["Subject"] = pd.to_numeric(sex["Subject"], errors="coerce")

    demographics = pd.merge(age, sex, on="Subject", how="inner")
    demographics = demographics[demographics["Subject"].isin(subject_ids)].copy()
    demographics = demographics.dropna(subset=["Age_in_Yrs", "Gender", "Subject"])
    demographics["Age_in_Yrs"] = pd.to_numeric(
        demographics["Age_in_Yrs"], errors="coerce"
    )
    demographics = demographics.dropna(subset=["Age_in_Yrs"])

    n_females = int((demographics["Gender"].str.upper() == "F").sum())
    mean_age = (
        float(demographics["Age_in_Yrs"].mean())
        if not demographics.empty
        else float("nan")
    )
    sd_age = (
        float(demographics["Age_in_Yrs"].std())
        if not demographics.empty
        else float("nan")
    )

    print(f"Number of females   : {n_females}")
    print(f"Mean age (all subs) : {mean_age:.2f} years")
    print(f"SD age (all subs) : {sd_age:.2f} years")


if __name__ == "__main__":
    main()
