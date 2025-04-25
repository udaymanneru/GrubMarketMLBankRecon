# === Optimized Multi-Pass Fuzzy/NLP Matcher with Vendor Grouping and Greedy Pass 3 ===
import pandas as pd
import math
import numpy as np
import spacy
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import re
from itertools import combinations

nlp = spacy.load("en_core_web_md")

# === Preprocessing ===
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"(merchant|pos)?\s*purchase\s*terminal\s*\d+", "", text)
    text = re.sub(r"seq\s*#\s*\d+", "", text)
    text = re.sub(r"[a-z]{2}\d{12,}", "", text)
    text = re.sub(r"\d{2,}", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    vendor_keywords = ["paypal", "indeed", "amazon", "office depot", "liberty mutual", "trugrid", "comcast", "savemart"]
    found = [v for v in vendor_keywords if v in text]
    if found:
        return " ".join(found)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def process_row(row):
    parts = [str(row[col]) for col in ["Date", "Transaction Type", "Check", "Description"] if pd.notnull(row[col])]
    return clean_text(" ".join(parts))

def get_vector(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(nlp.vocab.vectors_length)
    return nlp(text).vector

def compute_score(gl_row, bank_row, alpha=0.5):
    fuzzy = fuzz.token_sort_ratio(gl_row['clean'], bank_row['clean']) / 100
    vector = cosine_similarity([gl_row['vector']], [bank_row['vector']])[0][0]
    amount = 1 if abs(gl_row['Amount'] - bank_row['Amount']) < 0.01 else 0
    date_diff = abs((pd.to_datetime(gl_row['Date']) - pd.to_datetime(bank_row['Date'])).days)
    date_bonus = 0.1 if date_diff <= 1 else 0
    score = alpha * fuzzy + (1 - alpha) * vector + 0.3 * amount + date_bonus
    return score

def match_pass_1(gl_df, bank_df):
    matches = []
    used_gl = set()
    used_bank = set()
    for i, gl_row in gl_df.iterrows():
        for j, bank_row in bank_df.iterrows():
            if i in used_gl or j in used_bank:
                continue
            if (gl_row['Amount'] == bank_row['Amount'] and
                pd.to_datetime(gl_row['Date']) == pd.to_datetime(bank_row['Date']) and
                clean_text(gl_row['Description']) == clean_text(bank_row['Description'])):
                matches.append((i, [j], 1.3))
                used_gl.add(i)
                used_bank.add(j)
    return matches, used_gl, used_bank

def match_pass_2(gl_df, bank_df, used_gl, used_bank, threshold=0.85):
    matches = []
    for i, gl_row in gl_df.iterrows():
        if i in used_gl: continue
        best_score, best_j = 0, None
        for j, bank_row in bank_df.iterrows():
            if j in used_bank: continue
            score = compute_score(gl_row, bank_row)
            if score > best_score:
                best_score = score
                best_j = j
        if best_score >= threshold:
            matches.append((i, [best_j], best_score))
            used_gl.add(i)
            used_bank.add(best_j)
    return matches, used_gl, used_bank

def greedy_group_match(gl_row, candidate_group, tolerance=0.01):
    group = candidate_group.sort_values(by='Amount', ascending=False)
    selected, total = [], 0
    for idx, row in group.iterrows():
        selected.append(idx)
        total += row['Amount']
        if abs(total - gl_row['Amount']) < tolerance:
            return selected
        if total > gl_row['Amount']:
            return None
    return None

def match_pass_3(gl_df, bank_df, used_gl, used_bank, threshold=0.85):
    matches = []
    unmatched_gl = gl_df.loc[~gl_df.index.isin(used_gl)]
    unmatched_bank = bank_df.loc[~bank_df.index.isin(used_bank)]

    for i, gl_row in unmatched_gl.iterrows():
        # Step 1: Filter bank candidates by amount magnitude and sign
        candidates = unmatched_bank[
            (abs(unmatched_bank['Amount']) <= abs(gl_row['Amount']) + 0.01) &
            (unmatched_bank['Amount'].apply(lambda x: x * gl_row['Amount'] > 0)) &
            (pd.to_datetime(unmatched_bank['Date']) >= pd.to_datetime(gl_row['Date']))
        ].copy()

        if candidates.empty:
            continue

        # Step 2: Score and sort candidates by semantic similarity
        candidates.loc[:, 'score'] = candidates.apply(lambda row: compute_score(gl_row, row), axis=1)

        # Step 3: Group by date (client emphasized same-day matching)
        for date, date_group in candidates.groupby("Date"):
            # Step 4: If vendor info exists, group by 'clean'
            if date_group['clean'].notnull().any():
                grouped = date_group.groupby("clean")
            else:
                grouped = {'all': date_group}  # fallback to one group if no 'clean'

            # Step 5: Greedy match inside each group
            for _, group in grouped.items() if isinstance(grouped, dict) else grouped:
                group = group.sort_values(by='score', ascending=False)
                matched_indices = greedy_group_match(gl_row, group)

                if matched_indices:
                    avg_score = np.mean([compute_score(gl_row, bank_df.loc[j]) for j in matched_indices])
                    if avg_score >= threshold:
                        matches.append((i, matched_indices, avg_score))
                        used_gl.add(i)
                        used_bank.update(matched_indices)
                        break  # stop checking after first good match

            if i in used_gl:
                break  # GL matched in this date group, move on to next GL

    return matches, used_gl, used_bank


def setup_data(gl_data, bank_data):
    gl_df = pd.DataFrame(gl_data, columns=["Date", "Transaction Type", "Check", "Description", "Amount"])
    bank_df = pd.DataFrame(bank_data, columns=["Date", "Transaction Type", "Check", "Description", "Amount"])
    gl_df['clean'] = gl_df.apply(process_row, axis=1)
    bank_df['clean'] = bank_df.apply(process_row, axis=1)
    gl_df['vector'] = gl_df['clean'].apply(get_vector)
    bank_df['vector'] = bank_df['clean'].apply(get_vector)
    return gl_df, bank_df

def run_all_passes(gl_df, bank_df):
    match_1, used_gl, used_bank = match_pass_1(gl_df, bank_df)
    match_2, used_gl, used_bank = match_pass_2(gl_df, bank_df, used_gl, used_bank)
    match_3, used_gl, used_bank = match_pass_3(gl_df, bank_df, used_gl, used_bank)
    all_matches = match_1 + match_2 + match_3
    unmatched_gl = gl_df.loc[~gl_df.index.isin(used_gl)]
    unmatched_bank = bank_df.loc[~bank_df.index.isin(used_bank)]

    print("\n✅ MATCHED TRANSACTIONS:")
    for i, js, score in all_matches:
        gl_desc = gl_df.at[i, 'Description']
        bank_descs = ", ".join([str(bank_df.at[j, 'Description']) for j in js])
        print(f"GL: {gl_desc:<60} ↔ BANK: {bank_descs:<60} | Score: {score:.2f}")

    print("\n❌ UNMATCHED GL TRANSACTIONS:")
    for i, row in unmatched_gl.iterrows():
        print(row[['Date', 'Description', 'Amount']].to_dict())
        print("  Top Match Suggestions:")
        candidates = []
        for j, bank_row in unmatched_bank.iterrows():
            score = compute_score(row, bank_row)
            candidates.append((score, j))
        top = sorted(candidates, reverse=True)[:3]
        for score, j in top:
            print(f"    → BANK: {bank_df.at[j, 'Description']} | Date: {bank_df.at[j, 'Date']} | Amount: {bank_df.at[j, 'Amount']} | Score: {score:.2f}")

    print("\n❌ UNMATCHED BANK TRANSACTIONS:")
    for _, row in unmatched_bank.iterrows():
        print(row[['Date', 'Description', 'Amount']].to_dict())

# === Load & Run ===
gl_data = pd.read_excel("/Users/user/Desktop/school/scope/complexity.xlsx", sheet_name="GL")
bank_data = pd.read_excel("/Users/user/Desktop/school/scope/complexity.xlsx", sheet_name="Bank")
gl_df, bank_df = setup_data(gl_data, bank_data)
run_all_passes(gl_df, bank_df)
