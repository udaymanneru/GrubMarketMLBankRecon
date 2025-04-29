import pandas as pd
import numpy as np
import spacy
import re
from rapidfuzz import fuzz
import os
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")


# === Utilities ===
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return "[MISSING]"
    text = str(text).lower()
    text = re.sub(r"(merchant|pos)?\s*purchase\s*terminal\s*\d+", "", text)
    text = re.sub(r"seq\s*#\s*\d+", "", text)
    text = re.sub(r"[a-z]{2}\d{12,}", "", text)
    text = re.sub(r"\d{2,}", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_vector(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(nlp.vocab.vectors_length)
    return nlp(text).vector


# === Scoring Function ===
def compute_score(gl_row, bank_row):
    ref_gl = str(gl_row.get("Check_Ref", "")).lower()
    ref_bank = str(bank_row.get("Check_Ref", "")).lower()
    desc_gl = clean_text(gl_row.get("Description", ""))
    desc_bank = clean_text(bank_row.get("Description", ""))

    amount_score = 1.0 if gl_row["Amount"] == bank_row["Amount"] else 0.0
    ref_score = 0.0 if '[missing]' in (ref_gl, ref_bank) else fuzz.token_sort_ratio(ref_gl, ref_bank) / 100.0
    date_score = 1.0 if pd.to_datetime(gl_row["Date"]) == pd.to_datetime(bank_row["Date"]) else 0.0
    desc_score = 0.0 if '[missing]' in (desc_gl, desc_bank) else fuzz.token_sort_ratio(desc_gl, desc_bank) / 100.0

    # Cosine similarity between descriptions
    vec_gl = get_vector(desc_gl)
    vec_bank = get_vector(desc_bank)
    cos_score = cosine_similarity([vec_gl], [vec_bank])[0][0]

    amount_weight = 0.40
    ref_weight = 0.30
    date_weight = 0.15
    desc_weight = 0.15

    usable_weight = 0.0
    score_contribution = 0.0

    score_contribution += amount_weight * amount_score
    usable_weight += amount_weight

    if '[missing]' not in (ref_gl, ref_bank):
        score_contribution += ref_weight * ref_score
        usable_weight += ref_weight

    score_contribution += date_weight * date_score
    usable_weight += date_weight

    if '[missing]' not in (desc_gl, desc_bank):
        # blend fuzzy and cosine
        blended_desc_score = (desc_score + cos_score) / 2
        score_contribution += desc_weight * blended_desc_score
        usable_weight += desc_weight

    total_score = score_contribution / usable_weight if usable_weight > 0 else 0.0

    return total_score, amount_score, ref_score, date_score, desc_score


# === Matching Pass ===
def match_transactions(gl_df, bank_df, threshold=0.75):
    matches = []
    used_gl, used_bank = set(), set()

    for i, gl_row in gl_df.iterrows():
        best_score = -1
        best_j = None
        best_components = None

        for j, bank_row in bank_df.iterrows():
            if j in used_bank:
                continue
            score, amount_s, ref_s, date_s, desc_s = compute_score(gl_row, bank_row)
            if score > best_score:
                best_score = score
                best_j = j
                best_components = (amount_s, ref_s, date_s, desc_s)

        if best_score >= threshold:
            matches.append((i, best_j, best_score, best_components))
            used_gl.add(i)
            used_bank.add(best_j)

    unmatched_gl = gl_df[~gl_df.index.isin(used_gl)]
    unmatched_bank = bank_df[~bank_df.index.isin(used_bank)]

    return matches, unmatched_gl, unmatched_bank


# === Explanation Generator ===
def generate_explanation(score, components, threshold=0.75):
    amount_s, ref_s, date_s, desc_s = components

    if score >= threshold:
        return f"✅ Exact match. Score: {score:.4f}."

    # If amount and date match, need to focus on reference/description differences
    if amount_s == 1.0 and date_s == 1.0:
        issues = []

        if ref_s < 0.85 and (ref_s <= desc_s):
            issues.append(f"Reference number similarity is low ({ref_s:.2f}), recommend fuzzy matching techniques")
        if desc_s < 0.85 and (desc_s < ref_s):
            issues.append(f"Description similarity is low ({desc_s:.2f}), recommend NLP processing techniques")

        if ref_s < 0.85 and desc_s < 0.85 and abs(ref_s - desc_s) <= 0.05:
            # both are similarly bad
            issues = [
                f"Reference number similarity is low ({ref_s:.2f}), recommend fuzzy matching techniques",
                f"Description similarity is low ({desc_s:.2f}), recommend NLP processing techniques"
            ]

        if issues:
            explanation = "❌ No match. Issues: " + " and ".join(issues) + ". Manual review needed."
        else:
            explanation = "❌ No match. Issues: Amount/date match but unclear reason. Manual review needed."
    else:
        # Amount/date mismatch is more fundamental
        issues = ["Amount/date mismatch detected"]

        if ref_s < 0.85:
            issues.append(f"Reference number similarity is low ({ref_s:.2f}), recommend fuzzy matching techniques")
        if desc_s < 0.85:
            issues.append(f"Description similarity is low ({desc_s:.2f}), recommend NLP processing techniques")

        explanation = "❌ No match. Issues: " + ". ".join(issues) + ". Manual review needed."

    return explanation



# === Annotate DataFrames ===
def annotate(gl_df, bank_df, matches, unmatched_gl, unmatched_bank, threshold=0.75):
    gl_has_id = "ID" in gl_df.columns
    bank_has_id = "ID" in bank_df.columns

    gl_df["Match Explanation"] = ""
    gl_df["Match ID"] = ""
    bank_df["Match Explanation"] = ""
    bank_df["Match ID"] = ""

    # === Matched entries
    for i, j, score, components in matches:
        gl_id = gl_df.at[i, "ID"] if gl_has_id else i
        bank_id = bank_df.at[j, "ID"] if bank_has_id else j
        source = gl_df.at[i, "Source"] if "Source" in gl_df.columns else "GL"

        explanation = generate_explanation(score, components, threshold)

        gl_df.at[i, "Match Explanation"] = explanation
        gl_df.at[i, "Match ID"] = f"Matched to Bank ID={bank_id}"

        bank_df.at[j, "Match Explanation"] = explanation
        bank_df.at[j, "Match ID"] = f"Matched to {source}-ID={gl_id}"

    # === Unmatched GL entries with suggestions
    for i, gl_row in unmatched_gl.iterrows():
        candidates = []
        best_score = -1
        best_components = None

        for j, bank_row in unmatched_bank.iterrows():
            score, amount_s, ref_s, date_s, desc_s = compute_score(gl_row, bank_row)
            bank_id = bank_row["ID"] if bank_has_id else j
            candidates.append((score, bank_id))
            if score > best_score:
                best_score = score
                best_components = (amount_s, ref_s, date_s, desc_s)

        top3 = sorted(candidates, reverse=True)[:3]
        source = gl_df.at[i, "Source"] if "Source" in gl_df.columns else "GL"

        issue_explanation = generate_explanation(best_score, best_components, threshold)
        match_id_info = " | ".join(
            [
                f"Suggestion {idx}: Bank ID={bid} (Score={s:.2f}), Matched from {source}"
                for idx, (s, bid) in enumerate(top3, 1)
            ]
        )

        gl_df.at[i, "Match Explanation"] = issue_explanation
        gl_df.at[i, "Match ID"] = match_id_info


# === Full Run ===
def run_reconciliation(gl_file, bank_file, output_file="data/annotated_output.xlsx"):
    os.makedirs("data", exist_ok=True)  # ensure output directory exists

    gl_ext = os.path.splitext(gl_file)[1].lower()
    bank_ext = os.path.splitext(bank_file)[1].lower()

    if gl_ext == ".csv":
        gl_df = pd.read_csv(gl_file).rename(columns={"Memo/Description": "Description", "Ref #": "Check_Ref"})
    elif gl_ext in [".xls", ".xlsx"]:
        gl_df = pd.read_excel(gl_file)
    else:
        raise ValueError(f"Unsupported GL file format: {gl_ext}")

    if bank_ext == ".csv":
        bank_df = pd.read_csv(bank_file)
    elif bank_ext in [".xls", ".xlsx"]:
        bank_df = pd.read_excel(bank_file)
    else:
        raise ValueError(f"Unsupported Bank file format: {bank_ext}")

    matches, unmatched_gl, unmatched_bank = match_transactions(gl_df, bank_df)
    annotate(gl_df, bank_df, matches, unmatched_gl, unmatched_bank)

    print("\n✅ CONFIRMED MATCHES (score ≥ 0.75):")
    for i, j, score, components in matches:
        desc_gl = gl_df.at[i, 'Description'] if pd.notna(gl_df.at[i, 'Description']) else '[NO DESCRIPTION]'
        desc_bank = bank_df.at[j, 'Description'] if pd.notna(bank_df.at[j, 'Description']) else '[NO DESCRIPTION]'
        source = gl_df.at[i, "Source"] if "Source" in gl_df.columns else "GL"
        print(f"{source}: {desc_gl:<60} ↔ BANK: {desc_bank:<60} | Score: {score:.4f}")

    print("\n❌ UNMATCHED GL TRANSACTIONS:")
    for i, gl_row in unmatched_gl.iterrows():
        print(gl_row[['Date', 'Description', 'Amount']].to_dict())
        print("  Top Match Suggestions:")
        candidates = []
        for j, bank_row in unmatched_bank.iterrows():
            score, *_ = compute_score(gl_row, bank_row)
            candidates.append((score, j))
        top = sorted(candidates, reverse=True)[:3]
        for score, j in top:
            print(
                f"    → BANK: {bank_df.at[j, 'Description']} | Date: {bank_df.at[j, 'Date']} | Amount: {bank_df.at[j, 'Amount']} | Score: {score:.4f}"
            )

    print("\n❌ UNMATCHED BANK TRANSACTIONS:")
    for _, row in unmatched_bank.iterrows():
        print(row[['Date', 'Description', 'Amount']].to_dict())

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        gl_df.to_excel(writer, index=False, sheet_name="GL")
        bank_df.to_excel(writer, index=False, sheet_name="Bank")

    # Build 'Next Action' column for failed GL
    failed_gl = unmatched_gl.copy()
    failed_gl["Next Action"] = ""

    for idx, gl_row in failed_gl.iterrows():
        candidates = []
        for j, bank_row in unmatched_bank.iterrows():
            score, amount_s, ref_s, date_s, desc_s = compute_score(gl_row, bank_row)
            candidates.append((score, amount_s, ref_s, date_s, desc_s))

        if not candidates:
            action = "No suggestions available. Manual review needed."
        else:
            best = max(candidates, key=lambda x: x[0])
            _, amount_s, ref_s, date_s, desc_s = best

            if amount_s == 1.0 and date_s == 1.0:
                problems = []
                if ref_s < 0.85 and (ref_s <= desc_s):
                    problems.append("Focus on Ref (fuzzy)")
                if desc_s < 0.85 and (desc_s < ref_s):
                    problems.append("Focus on Desc (NLP)")
                if ref_s < 0.85 and desc_s < 0.85 and abs(ref_s - desc_s) <= 0.05:
                    problems = ["Focus on Ref (fuzzy)", "Focus on Desc (NLP)"]
                action = (
                    "Amount/date match. " + " and ".join(problems) + "."
                    if problems
                    else "Amount/date match but unclear issue. Manual review needed."
                )
            else:
                problems = []
                if ref_s < 0.85:
                    problems.append("Focus on Ref (fuzzy)")
                if desc_s < 0.85:
                    problems.append("Focus on Desc (NLP)")
                action = (
                    "Amount/date mismatch. " + " and ".join(problems) + "."
                    if problems
                    else "Amount/date mismatch. Manual review needed."
                )

        failed_gl.at[idx, "Next Action"] = action

    failed_gl = failed_gl.drop(columns=[col for col in ['clean', 'vector'] if col in failed_gl.columns], errors='ignore')
    failed_bank = unmatched_bank.drop(columns=[col for col in ['clean', 'vector'] if col in unmatched_bank.columns], errors='ignore')

    failed_gl.to_csv("data/failed_gl.csv", index=False)
    failed_bank.to_csv("data/failed_bank.csv", index=False)

    print("\n✅ Saved 'failed_gl.csv' and 'failed_bank.csv' with next action hints.")

# Example Usage:
run_reconciliation("data/cleaned_combined_records.csv", "data/Cleaned_Bank (1).csv")
