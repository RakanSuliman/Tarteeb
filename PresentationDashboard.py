import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# Try importing scikit-learn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        accuracy_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =========================
# Helpers
# =========================

def extract_year_from_name(col_name: str):
    """Extract a 3–4 digit year from a column name like 'avg_1442' or 'ksa_rank_1443'."""
    m = re.search(r"(\d{3,4})", col_name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="School Data Dashboard - Tarteeb",
    layout="wide",
)

st.title("Interactive School Data Dashboard")
st.caption("Use the filters on the left to explore school data, rankings, and averages per year.")

# =========================
# Load data
# =========================
with st.sidebar:
    st.header("Load data")
    uploaded_file = st.file_uploader(
        "Upload CSV file containing school data",
        type=["csv"]
    )

    st.markdown("---")
    st.header("General settings")

if uploaded_file is None:
    st.info("Please upload a CSV file from the sidebar to begin.")
    st.stop()

df = load_data(uploaded_file)

# Make sure expected columns exist (will crash early if not)
required_cols = [
    "year", "mno", "school_name_ar",
    "city_ar", "area_ar", "region_ar",
    "authority_ar", "education_type_ar",
    "exam_type_ar", "exam_special_ar",
    "lat", "lon"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in file: {missing}")
    st.stop()

# =========================
# Sidebar Filters
# =========================
with st.sidebar:
    st.header("Data filters")

    # Year filter (e.g. 1441, 1442, 1443, 1445...)
    years = [1441, 1442, 1443, 1444, 1445]
    selected_years = st.multiselect(
        "Hijri year",
        options=years,
        default=years  # show all by default
    )

    regions = sorted(df["region_ar"].dropna().unique())
    selected_regions = st.multiselect(
        "Region",
        options=regions,
        default=regions
    )

    cities = sorted(df["city_ar"].dropna().unique())
    selected_cities = st.multiselect(
        "City",
        options=cities,
        default=cities
    )

    authorities = sorted(df["authority_ar"].dropna().unique())
    selected_authorities = st.multiselect(
        "Authority type (government/private...)",
        options=authorities,
        default=authorities
    )

    edu_types = sorted(df["education_type_ar"].dropna().unique())
    selected_edu_types = st.multiselect(
        "Education type",
        options=edu_types,
        default=edu_types
    )

    exam_types = sorted(df["exam_type_ar"].dropna().unique())
    selected_exam_types = st.multiselect(
        "Exam type",
        options=exam_types,
        default=exam_types
    )

    exam_special = sorted(df["exam_special_ar"].dropna().unique())
    selected_exam_special = st.multiselect(
        "Specialization (science/literature...)",
        options=exam_special,
        default=exam_special
    )

# =========================
# Apply filters
# =========================
filtered = df.copy()

if selected_years:
    filtered = filtered[filtered["year"].isin(selected_years)]

if selected_regions:
    filtered = filtered[filtered["region_ar"].isin(selected_regions)]

if selected_cities:
    filtered = filtered[filtered["city_ar"].isin(selected_cities)]

if selected_authorities:
    filtered = filtered[filtered["authority_ar"].isin(selected_authorities)]

if selected_edu_types:
    filtered = filtered[filtered["education_type_ar"].isin(selected_edu_types)]

if selected_exam_types:
    filtered = filtered[filtered["exam_type_ar"].isin(selected_exam_types)]

if selected_exam_special:
    filtered = filtered[filtered["exam_special_ar"].isin(selected_exam_special)]

if filtered.empty:
    st.warning("No data matches the current filters. Change the filters and try again.")
    st.stop()

st.subheader("Filtered data overview")
st.write(f"Number of rows: **{len(filtered)}**")
st.dataframe(filtered.head(20))

# =========================
# Score / Rank column selection
# =========================

# Auto-detect score and rank columns
score_cols = [c for c in df.columns if c.startswith("avg")]
rank_cols = [c for c in df.columns if "ksa_rank" in c]

default_score_col = score_cols[0] if score_cols else None
default_rank_col = "ksa_rank" if "ksa_rank" in df.columns else (rank_cols[0] if rank_cols else None)

col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)

with col_sc1:
    score_col = st.selectbox(
        "Score column used for analysis (e.g. avg_1442):",
        options=score_cols if score_cols else ["(no avg_... columns found)"],
        index=0 if score_cols else 0,
    )

with col_sc2:
    rank_col = st.selectbox(
        "National rank column:",
        options=rank_cols if rank_cols else ["(no ksa_rank... columns found)"],
        index=rank_cols.index(default_rank_col) if (default_rank_col and default_rank_col in rank_cols) else 0
    )

# Build exam-type options from current filtered data
exam_type_options = ["(all exam types)"]
if "exam_type_ar" in filtered.columns:
    exam_type_options += sorted(filtered["exam_type_ar"].dropna().unique().tolist())

gender_options = ["(all genders)"]
if "gender_code" in filtered.columns:
    gender_options += sorted(filtered["gender_code"].dropna().unique().tolist())

with col_sc3:
    selected_exam_type = st.selectbox(
        "Exam type:",
        options=exam_type_options,
        index=0,
    )

with col_sc4:
    selected_gender = st.selectbox(
        "Gender:",
        options=gender_options,
        index=0,
    )

if score_cols and score_col not in filtered.columns:
    st.error(f"Selected score column {score_col} is not present in the filtered data.")
    st.stop()

if rank_cols and rank_col not in filtered.columns:
    st.error(f"Selected rank column {rank_col} is not present in the filtered data.")
    st.stop()

if selected_exam_type != "(all exam types)":
    filtered = filtered[filtered["exam_type_ar"] == selected_exam_type]

if selected_gender != "(all genders)":
    filtered = filtered[filtered["gender_code"] == selected_gender]

score_year = extract_year_from_name(score_col) if score_cols else None
rank_year = extract_year_from_name(rank_col) if rank_cols else None

# =========================
# KPIs
# =========================
st.markdown("### Key metrics (KPIs)")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Number of schools (rows)", f"{len(filtered):,}")

with kpi2:
    if score_cols:
        mean_score = filtered[score_col].mean()
        st.metric(f"Mean of {score_col}", f"{mean_score:.2f}")
    else:
        st.metric("No score column", "-")

with kpi3:
    if rank_cols:
        best_rank = int(filtered[rank_col].min())
        st.metric("Best rank (smallest value)", f"{best_rank}")
    else:
        st.metric("No rank column", "-")

with kpi4:
    if rank_cols:
        worst_rank = int(filtered[rank_col].max())
        st.metric("Worst rank (largest value)", f"{worst_rank}")
    else:
        st.metric("No rank column", "-")

# =========================
# Charts
# =========================
st.markdown("### Charts")

top_n = st.slider("Number of top schools to show (by score)", 5, 50, 10, step=5)

# Bar chart: top N by score
if score_cols:
    top_schools = filtered.dropna(subset=[score_col]).nlargest(top_n, score_col)

    bar_chart = alt.Chart(top_schools).mark_bar().encode(
        x=alt.X(score_col, title=score_col),
        y=alt.Y("school_name_ar:N", sort='-x', title="School name"),
        tooltip=["school_name_ar", score_col, rank_col if rank_cols else "year"]
    ).properties(
        title=f"Top {top_n} schools by {score_col}",
        height=400
    )

    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.info("No score column (avg_...) found to build a chart.")

# Scatter: score vs rank
if score_cols and rank_cols:
    scatter = alt.Chart(filtered.dropna(subset=[score_col, rank_col])).mark_circle(size=60).encode(
        x=alt.X(rank_col, title=rank_col, scale=alt.Scale(reverse=True)),  # rank 1 is best
        y=alt.Y(score_col, title=score_col),
        color="region_ar:N",
        tooltip=["school_name_ar", "region_ar", "city_ar", score_col, rank_col]
    ).properties(
        title=f"{score_col} vs {rank_col}",
        height=400
    ).interactive()

    st.altair_chart(scatter, use_container_width=True)

# =========================
# Map of schools
# =========================
st.markdown("### Schools map")

map_df = filtered.dropna(subset=["lat", "lon"])[["school_name_ar", "city_ar", "region_ar", "lat", "lon"]]

if map_df.empty:
    st.info("There are not enough latitude/longitude values to display the map.")
else:
    st.map(
        map_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
        zoom=6
    )

# =========================
# School-level detail
# =========================
st.markdown("### School details")

school_names = filtered["school_name_ar"].dropna().unique()
selected_school = st.selectbox(
    "Select a school:",
    options=sorted(school_names)
)

school_detail = filtered[filtered["school_name_ar"] == selected_school]

st.write(f"Details for: **{selected_school}**")
st.dataframe(school_detail)

# =========================
# ML Section: Regression & Classification
# =========================

st.markdown("## Machine learning analysis (hypotheses)")

if not SKLEARN_AVAILABLE:
    st.warning(
        "scikit-learn is not installed. To enable the machine learning section, install it with:\n"
        "`pip install scikit-learn`"
    )
else:
    if filtered.shape[0] < 50:
        st.info("Warning: number of rows after filtering is small. Model results may be unstable.")

    tabs = st.tabs([
        "Regression: predict score",
        "Classification: top 10% schools",
        "Logistic: underperforming vs area avg",
        "Unsupervised: PCA of performance",
    ])

    # Common categorical features
    categorical_features = [
        c for c in [
            "region_ar", "city_ar", "authority_ar",
            "education_type_ar", "exam_type_ar", "exam_special_ar"
        ]
        if c in filtered.columns
    ]

    # =========================
    # TAB 1: REGRESSION (Hypothesis 1)
    # =========================
    with tabs[0]:
        st.subheader("Regression model to predict school average score")

        if not score_cols:
            st.warning("No score columns (avg_...) found to build a regression model.")
        else:
            # Build numeric features using ONLY previous years (compared to score_year)
            numeric_features_reg = []

            # Previous avg_* columns
            for c in score_cols:
                if c == score_col:
                    continue
                cy = extract_year_from_name(c)
                if score_year is None or (cy is not None and cy < score_year):   # no = sign to avoid leakage
                    numeric_features_reg.append(c)

            # Previous std_count_* columns
            for c in df.columns:
                if c.startswith("std_count_"):
                    cy = extract_year_from_name(c)
                    if score_year is None or (cy is not None and cy <= score_year):
                        numeric_features_reg.append(c)

            # Previous ksa_rank_* columns (not the current target rank column)
            for c in rank_cols:
                if c == rank_col:
                    continue
                cy = extract_year_from_name(c)
                if score_year is None or (cy is not None and cy < score_year):  # no = sign to avoid leakage
                    numeric_features_reg.append(c)

            # Keep only numeric and present in filtered
            numeric_features_reg = [
                c for c in sorted(set(numeric_features_reg))
                if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c])
            ]

            st.write("Numeric features used (previous years only):", numeric_features_reg)
            st.write("Categorical features used:", categorical_features)

            if not numeric_features_reg:
                st.warning("There are no suitable numeric features from previous years for regression.")
            else:
                run_reg = st.button("Run regression model", key="run_reg")

                if run_reg:
                    cols_needed = [score_col] + numeric_features_reg + categorical_features
                    data_reg = filtered.dropna(subset=cols_needed).copy()

                    if data_reg.shape[0] < 30:
                        st.warning("After dropping missing values, fewer than 30 rows remain. Results may be weak.")
                    else:
                        X = data_reg[numeric_features_reg + categorical_features]
                        y = data_reg[score_col].values

                        # Train/test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        # One-hot encode categoricals using train only
                        X_train = pd.get_dummies(
                            X_train,
                            columns=categorical_features,
                            drop_first=True
                        )
                        X_test = pd.get_dummies(
                            X_test,
                            columns=categorical_features,
                            drop_first=True
                        )
                        # Align columns between train and test
                        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

                        model = LinearRegression()

                        with st.spinner("Training regression model..."):
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        baseline_pred = np.full_like(y_test, y_train.mean())
                        baseline_mae = mean_absolute_error(y_test, baseline_pred)

                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("R² on test set", f"{r2:.3f}")
                        with m2:
                            st.metric("Mean absolute error (MAE)", f"{mae:.2f}")
                        with m3:
                            st.metric("Baseline MAE (predict train mean)", f"{baseline_mae:.2f}")

                        # Scatter plot: actual vs predicted
                        results_df = pd.DataFrame(
                            {"actual": y_test, "predicted": y_pred}
                        )

                        st.markdown("#### Actual vs predicted scores")

                        reg_scatter = alt.Chart(results_df.reset_index(drop=True)).mark_circle(size=60).encode(
                            x=alt.X("actual", title="Actual"),
                            y=alt.Y("predicted", title="Predicted"),
                            tooltip=["actual", "predicted"],
                        ).properties(
                            height=400,
                            title="Actual vs predicted"
                        )

                        line = alt.Chart(
                            pd.DataFrame({"x": [results_df.actual.min(), results_df.actual.max()]})
                        ).mark_line().encode(
                            x="x",
                            y="x",
                        )

                        st.altair_chart(reg_scatter + line, use_container_width=True)

    # =========================
    # TAB 2: CLASSIFICATION (Hypothesis 2)
    # =========================
    with tabs[1]:
        st.subheader("Classification model to predict top 10% schools")

        if not rank_cols:
            st.warning("No rank columns (ksa_rank...) found to build a classification model.")
        else:
            rank_series = filtered[rank_col].dropna()
            if rank_series.empty:
                st.warning("No valid values in the selected rank column.")
            else:
                threshold = np.percentile(rank_series, 10)  # smaller rank = better
                st.write(f"Top 10% threshold (best ranks) is approximately rank ≤ {threshold:.0f}")

                data_clf = filtered.dropna(subset=[rank_col]).copy()
                data_clf["is_top"] = (data_clf[rank_col] <= threshold).astype(int)

                if data_clf["is_top"].sum() < 5:
                    st.warning("Fewer than 5 top schools (positive class) after filtering. Model will not be useful.")
                elif data_clf["is_top"].nunique() < 2:
                    st.warning("There is only one class present in the data. Classification is not possible.")
                else:
                    # Build numeric features using ONLY previous years (compared to rank_year)
                    numeric_features_clf = []

                    # Previous avg_* columns
                    for c in score_cols:
                        cy = extract_year_from_name(c)
                        if rank_year is None or (cy is not None and cy < rank_year):
                            numeric_features_clf.append(c)

                    # Previous std_count_* columns
                    for c in df.columns:
                        if c.startswith("std_count_"):
                            cy = extract_year_from_name(c)
                            if rank_year is None or (cy is not None and cy < rank_year):
                                numeric_features_clf.append(c)

                    # Previous ksa_rank_* columns (not the current one)
                    for c in rank_cols:
                        if c == rank_col:
                            continue
                        cy = extract_year_from_name(c)
                        if rank_year is None or (cy is not None and cy < rank_year):
                            numeric_features_clf.append(c)

                    numeric_features_clf = [
                        c for c in sorted(set(numeric_features_clf))
                        if c in data_clf.columns and pd.api.types.is_numeric_dtype(data_clf[c])
                    ]
                    # avoid leaking a generic "avg" column if it slipped in
                    if "avg" in numeric_features_clf:
                        numeric_features_clf.remove("avg")

                    st.write("Numeric features used (previous years only):", numeric_features_clf)
                    st.write("Categorical features used:", categorical_features)

                    if not numeric_features_clf:
                        st.warning("There are no suitable numeric features from previous years for classification.")
                    else:
                        cols_needed = numeric_features_clf + categorical_features + ["is_top"]
                        data_clf = data_clf.dropna(subset=numeric_features_clf + categorical_features).copy()

                        X_clf = data_clf[numeric_features_clf + categorical_features]
                        y_clf = data_clf["is_top"].values

                        if X_clf.shape[0] < 50:
                            st.warning("After dropping missing values, fewer than 50 rows remain. Results may be unstable.")

                        run_clf = st.button("Run classification model", key="run_clf")

                        if run_clf:
                            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                                X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
                            )

                            # One-hot encode categoricals using train only
                            X_train_c = pd.get_dummies(
                                X_train_c,
                                columns=categorical_features,
                                drop_first=True
                            )
                            X_test_c = pd.get_dummies(
                                X_test_c,
                                columns=categorical_features,
                                drop_first=True
                            )
                            X_test_c = X_test_c.reindex(columns=X_train_c.columns, fill_value=0)

                            clf = LogisticRegression(max_iter=1000)

                            with st.spinner("Training classification model..."):
                                clf.fit(X_train_c, y_train_c)
                                y_pred_c = clf.predict(X_test_c)

                            acc = accuracy_score(y_test_c, y_pred_c)
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_test_c, y_pred_c, average="binary"
                            )

                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                st.metric("Accuracy", f"{acc:.3f}")
                            with b2:
                                st.metric("Precision (positive class)", f"{precision:.3f}")
                            with b3:
                                st.metric("Recall (positive class)", f"{recall:.3f}")
                            with b4:
                                st.metric("F1-score", f"{f1:.3f}")

                            st.markdown("#### Confusion matrix")

                            cm = pd.crosstab(
                                y_test_c,
                                y_pred_c,
                                rownames=["True label"],
                                colnames=["Predicted label"],
                                dropna=False,
                            )

                            st.table(cm)

    # =========================
    # TAB 3: Logistic regression (underperforming vs area_avg)
    # =========================
    with tabs[2]:
        st.subheader("Logistic regression: underperforming vs area average")

        needed_cols = [
            "avg", "area_avg",
            "avg_1441", "avg_1442",
            "ksa_rank_1441", "ksa_rank_1442",
            "std_count_1441", "std_count_1442",
        ]
        missing_lr = [c for c in needed_cols if c not in filtered.columns]

        if missing_lr:
            st.warning(f"The following required columns are missing for this model: {missing_lr}")
        else:
            data_lr = filtered.dropna(subset=needed_cols).copy()
            if data_lr.empty:
                st.warning("No rows left after dropping missing values for the logistic regression model.")
            else:
                # Target: 1 if school is underperforming compared to its area average
                data_lr["underperforming"] = (data_lr["avg"] < data_lr["area_avg"]).astype(int)

                st.write("Class balance (0 = not underperforming, 1 = underperforming):")
                st.write(data_lr["underperforming"].value_counts())

                features_lr = [
                    "avg_1441", "avg_1442",
                    "ksa_rank_1441", "ksa_rank_1442",
                    "std_count_1441", "std_count_1442",
                ]

                X_lr = data_lr[features_lr]
                y_lr = data_lr["underperforming"].values

                if y_lr.sum() == 0 or y_lr.sum() == len(y_lr):
                    st.warning("Only one class present in the data after filtering. Logistic regression is not possible.")
                else:
                    run_lr = st.button("Run underperforming logistic model", key="run_lr")

                    if run_lr:
                        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
                            X_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr
                        )

                        log_reg = LogisticRegression(max_iter=1000)

                        with st.spinner("Training logistic regression model..."):
                            log_reg.fit(X_train_lr, y_train_lr)
                            y_pred_lr = log_reg.predict(X_test_lr)
                            y_probs_lr = log_reg.predict_proba(X_test_lr)[:, 1]

                        acc_lr = accuracy_score(y_test_lr, y_pred_lr)
                        prec_lr = precision_score(y_test_lr, y_pred_lr)
                        rec_lr = recall_score(y_test_lr, y_pred_lr)
                        f1_lr = f1_score(y_test_lr, y_pred_lr)

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Accuracy", f"{acc_lr:.3f}")
                        with c2:
                            st.metric("Precision", f"{prec_lr:.3f}")
                        with c3:
                            st.metric("Recall", f"{rec_lr:.3f}")
                        with c4:
                            st.metric("F1-score", f"{f1_lr:.3f}")

                        st.markdown("#### Confusion matrix")

                        cm_lr = pd.crosstab(
                            y_test_lr,
                            y_pred_lr,
                            rownames=["True label"],
                            colnames=["Predicted label"],
                            dropna=False,
                        )
                        st.table(cm_lr)

                        # ROC curve
                        fpr, tpr, thresholds = roc_curve(y_test_lr, y_probs_lr)
                        auc = roc_auc_score(y_test_lr, y_probs_lr)

                        st.markdown("#### ROC curve")

                        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                        roc_chart = alt.Chart(roc_df).mark_line().encode(
                            x=alt.X("fpr", title="False Positive Rate"),
                            y=alt.Y("tpr", title="True Positive Rate"),
                        ).properties(
                            height=400,
                            title=f"ROC curve (AUC = {auc:.3f})"
                        )

                        diag_df = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
                        diag_chart = alt.Chart(diag_df).mark_line(strokeDash=[5, 5]).encode(
                            x="fpr",
                            y="tpr",
                        )

                        st.altair_chart(roc_chart + diag_chart, use_container_width=True)

                        st.markdown("#### Example predicted probabilities (first 20 rows)")
                        probs_preview = pd.DataFrame({
                            "true_label": y_test_lr,
                            "predicted_label": y_pred_lr,
                            "prob_underperforming": y_probs_lr,
                        }).reset_index(drop=True).head(20)
                        st.dataframe(probs_preview)

    # =========================
    # TAB 4: Unsupervised PCA (and clustering hooks)
    # =========================
    with tabs[3]:
        st.subheader("Unsupervised learning: PCA of performance features")

        cluster_features = ["avg", "dept_avg", "office_avg", "area_avg"]
        missing_cluster = [c for c in cluster_features if c not in filtered.columns]

        if missing_cluster:
            st.warning(f"Cannot run PCA. Missing columns: {missing_cluster}")
        else:
            x = filtered[cluster_features].dropna()
            if x.empty:
                st.warning("No rows with complete performance features for PCA.")
            else:
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(x)

                pca = PCA(n_components=2)
                x_pca = pca.fit_transform(x_scaled)

                expl = pca.explained_variance_ratio_
                total_expl = expl.sum()

                st.write("Explained variance ratio (PC1, PC2):", expl)
                st.write(f"Total variance explained by first 2 PCs: **{total_expl:.3f}**")

                # Build a DataFrame for plotting
                pca_df = pd.DataFrame({
                    "PC1": x_pca[:, 0],
                    "PC2": x_pca[:, 1],
                    "avg": x["avg"].values,
                    "dept_avg": x["dept_avg"].values,
                }, index=x.index)

                # Add region/city if available for coloring
                if "region_ar" in filtered.columns:
                    pca_df["region_ar"] = filtered.loc[pca_df.index, "region_ar"]
                    color_field = "region_ar:N"
                else:
                    color_field = alt.value("steelblue")

                st.markdown("#### PCA scatter plot (PC1 vs PC2)")

                pca_chart = alt.Chart(pca_df.reset_index(drop=True)).mark_circle(size=60, opacity=0.6).encode(
                    x=alt.X("PC1", title="PC1"),
                    y=alt.Y("PC2", title="PC2"),
                    color=color_field,
                    tooltip=["PC1", "PC2", "avg", "dept_avg"] + (["region_ar"] if "region_ar" in pca_df.columns else []),
                ).properties(
                    height=400,
                    title="PCA of performance features"
                )

                st.altair_chart(pca_chart, use_container_width=True)

                st.info(
                    "You can extend this section later by running K-means (or other clustering) on the `PC1` and `PC2` "
                    "coordinates to group schools with similar performance profiles."
                )
