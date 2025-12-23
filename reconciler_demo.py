import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Reconcile AI Demo", layout="wide")

st.title("Reconcile AI")
st.markdown("**Free open-source tool to compare two CSVs — find mismatches, anomalies, and more in seconds.**")

@st.cache_data
def get_sample_data():
    df1 = pd.DataFrame({
        'OrderID': ['ORD001', 'ORD001', 'ORD002', 'ORD003', 'ORD004'],
        'Revenue': [5000.25, 7500.50, 2000.75, 3000.00, None],
        'CloseDate': ['2025-01-01', '2025-01-02', '2025-01-02', '2025-01-03', '2025-01-04'],
        'ClientCode': ['C001', 'C002', 'C002', 'C003', 'C004'],
        'Notes': ['Contract signed', 'Pending approval', '', 'Urgent', 'Incomplete']
    })
    df2 = pd.DataFrame({
        'OrderID': ['ORD001', 'ORD001', 'ORD002', 'ORD005', 'ORD006'],
        'Revenue': [5000.26, 7500.50, 2000.75, 4000.00, 10000.00],
        'CloseDate': ['2025-01-01', '2025-01-02', '2025-01-02', '2025-01-05', ''],
        'ClientCode': ['C001', 'C002', 'C002', 'C005', 'C006'],
        'Notes': ['Signed contract', 'Approved', 'Note2', 'New client', 'High value']
    })
    return df1, df2

use_sample = st.checkbox("Try with Sample Data (recommended)", value=True)

if use_sample:
    df1, df2 = get_sample_data()
    st.success("Sample data loaded!")
else:
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("First CSV", type="csv")
    with col2:
        file2 = st.file_uploader("Second CSV", type="csv")
    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

if 'df1' in locals():
    col1, col2 = st.columns(2)
    with col1: st.write("**File 1**"); st.dataframe(df1.head())
    with col2: st.write("**File 2**"); st.dataframe(df2.head())

    if st.button("Run Reconciliation", type="primary"):
        # Key detection
        common = df1.columns.intersection(df2.columns).tolist()
        keys = []
        for c in common:
            if df1[c].nunique() == len(df1) and df2[c].nunique() == len(df2):
                keys.append([c])
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                k1 = df1[[common[i], common[j]]].astype(str).agg('|'.join, axis=1)
                k2 = df2[[common[i], common[j]]].astype(str).agg('|'.join, axis=1)
                if k1.nunique() == len(k1) and k2.nunique() == len(k2):
                    keys.append([common[i], common[j]])
                    break
        suggested = keys[0] if keys else common[:1]
        key_cols = st.multiselect("Key columns", common, default=suggested)

        if key_cols:
            df1['key'] = df1[key_cols].astype(str).agg('|'.join, axis=1)
            df2['key'] = df2[key_cols].astype(str).agg('|'.join, axis=1)
            merged = df1.merge(df2, on='key', how='outer', suffixes=('_1', '_2'))

            # Anomaly detection
            st.subheader("Anomaly Detection")
            num_cols = [c for c in common if pd.api.types.is_numeric_dtype(df1[c])]
            if num_cols:
                col = st.selectbox("Find outliers in", num_cols)
                data = pd.concat([df1[[col]], df2[[col]]]).dropna()
                if len(data) > 10:
                    iso = IsolationForest(contamination=0.1)
                    preds = iso.fit_predict(data)
                    anomalies = data[preds == -1]
                    if len(anomalies):
                        st.error(f"{len(anomalies)} anomalies found")
                        st.dataframe(anomalies)
                    else:
                        st.success("No anomalies")
                else:
                    st.info("Not enough data")

            # Summary
            st.subheader("Reconciliation Summary")
            non_key = [c for c in common if c not in key_cols]
            summary = []
            for c in non_key:
                v1 = merged[f'{c}_1']
                v2 = merged[f'{c}_2']
                summary.append({
                    "Column": c,
                    "Matches": (v1 == v2).sum(),
                    "Mismatches": ((v1.notnull()) & (v2.notnull()) & (v1 != v2)).sum(),
                    "Only File 1": (v1.notnull() & v2.isnull()).sum(),
                    "Only File 2": (v2.notnull() & v1.isnull()).sum()
                })
            st.dataframe(pd.DataFrame(summary))
            st.download_button("Download Summary", pd.DataFrame(summary).to_csv(index=True), "summary.csv")

st.caption("Reconcile AI — Free Open Source Demo | Built with ❤️ using Streamlit")
