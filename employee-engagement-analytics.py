import os
import sys
import numpy as np
import pandas as pd
import random
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from faker import Faker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the entire pipeline."""
    SEED = 42
    N_EMPLOYEES = 350
    DEPARTMENTS = ["Engineering", "Data", "Finance", "Operations", "Sales", "HR", "Marketing", "Support"]
    DEPT_PROBS = [.23, .12, .12, .16, .17, .06, .08, .06]
    LOCATIONS = ["Dublin", "Cork", "Galway", "Remote-IE"]
    LOCATION_PROBS = [.55, .15, .1, .2]
    WORK_MODELS = ["Hybrid", "Remote", "Office"]
    WORK_MODEL_PROBS = [.6, .25, .15]
    SURVEY_START = "2023-10"
    SURVEY_END = "2025-09"
    
    # Engagement scoring weights
    SURVEY_WEIGHT = 0.7
    SENTIMENT_WEIGHT = 0.3
    
    # Risk model weights
    RISK_WEIGHTS = {
        "engagement": 0.55,
        "tenure": 0.25,
        "span": 0.15,
        "remote": 0.05
    }
    
    # Department engagement baselines (Likert 1-5 scale)
    DEPT_BIAS = {
        "Engineering":   {"alignment": 3.7, "workload": 3.2, "recognition": 3.2, "growth": 3.5},
        "Data":          {"alignment": 3.8, "workload": 3.3, "recognition": 3.3, "growth": 3.7},
        "Finance":       {"alignment": 3.6, "workload": 3.1, "recognition": 3.2, "growth": 3.3},
        "Operations":    {"alignment": 3.4, "workload": 3.0, "recognition": 3.0, "growth": 3.2},
        "Sales":         {"alignment": 3.5, "workload": 3.1, "recognition": 3.4, "growth": 3.3},
        "HR":            {"alignment": 3.7, "workload": 3.4, "recognition": 3.6, "growth": 3.4},
        "Marketing":     {"alignment": 3.6, "workload": 3.2, "recognition": 3.3, "growth": 3.4},
        "Support":       {"alignment": 3.3, "workload": 2.9, "recognition": 3.0, "growth": 3.1},
    }
    
    # Comment templates by sentiment
    COMMENTS = {
        "positive": [
            "Supportive team and flexible hours",
            "Great manager and clear goals",
            "Good growth opportunities and training",
            "Collaborative culture, love the projects",
            "Recognized for my work recently",
            "Strong leadership and transparent communication",
            "Excellent work-life balance here",
            "Challenging work with great autonomy",
            "Team celebrates wins together",
            "Manager provides regular feedback"
        ],
        "neutral": [
            "Workload is manageable most weeks",
            "Processes are okay but could be improved",
            "Communication is fine, some delays",
            "Hybrid schedule works most of the time",
            "Tools are adequate for my tasks",
            "Standard benefits and compensation",
            "Some projects more interesting than others",
            "Getting used to new systems",
            "Team dynamic is stable"
        ],
        "negative": [
            "Too many meetings and unclear priorities",
            "Heavy workload, burnout risk",
            "Poor recognition and slow decisions",
            "Lack of growth opportunities",
            "Delivery delays and context switching",
            "Limited feedback from management",
            "Unclear career progression path",
            "Resource constraints affecting quality",
            "Frequent reorganizations causing confusion",
            "Struggling with work-life balance"
        ]
    }

# Set random seeds for reproducibility
np.random.seed(Config.SEED)
random.seed(Config.SEED)
fake = Faker()
Faker.seed(Config.SEED)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_step(message):
    """Print step message."""
    print(f"\n→ {message}")

def print_success(message):
    """Print success message."""
    print(f"  ✓ {message}")

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs("data", exist_ok=True)

def likert(mu, sigma=0.9):
    """Generate Likert scale response (1-5)."""
    return int(np.clip(np.round(np.random.normal(mu, sigma)), 1, 5))

def sample_comment(avg_score):
    """Select comment based on average survey score."""
    if avg_score >= 4:
        return random.choice(Config.COMMENTS["positive"])
    elif avg_score >= 3:
        return random.choice(Config.COMMENTS["neutral"])
    else:
        return random.choice(Config.COMMENTS["negative"])

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_employees():
    """Generate employee dimension table."""
    print_step("Generating employee dimension table...")
    
    employees = []
    for i in range(1, Config.N_EMPLOYEES + 1):
        dept = np.random.choice(Config.DEPARTMENTS, p=Config.DEPT_PROBS)
        
        # Tech departments have slightly longer tenure
        base_tenure = 24 if dept in ["Engineering", "Data"] else 18
        tenure_m = max(1, int(np.random.normal(base_tenure, 10)))
        
        employees.append({
            "employee_id": i,
            "name": fake.name(),
            "age": np.random.randint(22, 58),
            "gender": np.random.choice(["M", "F", "Other"], p=[.48, .48, .04]),
            "dept": dept,
            "manager_id": np.random.randint(1, 40),
            "location": np.random.choice(Config.LOCATIONS, p=Config.LOCATION_PROBS),
            "hire_date": (datetime(2020, 1, 1) + relativedelta(months=-tenure_m)).date(),
            "tenure_months": tenure_m,
            "work_model": np.random.choice(Config.WORK_MODELS, p=Config.WORK_MODEL_PROBS),
            "fte": np.random.choice([1.0, 0.8, 0.6], p=[.82, .12, .06]),
        })
    
    df = pd.DataFrame(employees)
    print_success(f"Generated {len(df)} employees")
    return df

def generate_pulse_surveys(dim_employee):
    """Generate monthly pulse survey responses."""
    months = pd.period_range(Config.SURVEY_START, Config.SURVEY_END, freq="M").astype(str)
    print_step(f"Generating pulse surveys for {len(months)} months...")
    
    pulse_rows = []
    
    for _, emp in dim_employee.iterrows():
        dept_base = Config.DEPT_BIAS[emp.dept]
        wm_adjust = {"Hybrid": 0.1, "Remote": -0.1, "Office": 0.05}
        wm_bump = wm_adjust[emp.work_model]
        
        for month in months:
            # Generate survey responses with dept and work model effects
            q_alignment = likert(dept_base["alignment"] + wm_bump)
            q_workload = likert(dept_base["workload"] - (0.1 if emp.work_model == "Remote" else 0))
            q_recognition = likert(dept_base["recognition"] + (0.05 if emp.work_model != "Remote" else -0.05))
            q_growth = likert(dept_base["growth"] + np.random.normal(0, 0.2))
            
            # Generate contextual comment
            avg_score = np.mean([q_alignment, q_workload, q_recognition, q_growth])
            comment = sample_comment(avg_score)
            
            pulse_rows.append({
                "employee_id": emp.employee_id,
                "month": month,
                "q_workload": q_workload,
                "q_alignment": q_alignment,
                "q_recognition": q_recognition,
                "q_growth": q_growth,
                "comment_text": comment
            })
    
    df = pd.DataFrame(pulse_rows)
    print_success(f"Generated {len(df)} pulse responses")
    return df

# ============================================================================
# NLP & FEATURE ENGINEERING
# ============================================================================

def analyze_sentiment(df):
    """Perform sentiment analysis on comments."""
    print_step("Analyzing sentiment in comments...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return analyzer.polarity_scores(text)["compound"]
    
    df["sentiment_compound"] = df["comment_text"].apply(get_sentiment)
    print_success(f"Processed {len(df)} comments")
    print(f"  Mean sentiment: {df['sentiment_compound'].mean():.3f}")
    return df

def calculate_engagement_scores(df):
    """Calculate composite engagement score from survey + sentiment."""
    print_step("Computing composite engagement score...")
    
    # Survey mean (across 4 questions)
    survey_cols = ["q_workload", "q_alignment", "q_recognition", "q_growth"]
    df["survey_mean"] = df[survey_cols].mean(axis=1)
    
    # Normalize to 0-1 scale
    survey_scaled = (df["survey_mean"] - 1) / 4  # 1-5 → 0-1
    sentiment_scaled = (df["sentiment_compound"] + 1) / 2  # -1 to 1 → 0-1
    
    # Weighted blend
    engagement = (
        Config.SURVEY_WEIGHT * survey_scaled + 
        Config.SENTIMENT_WEIGHT * sentiment_scaled
    )
    df["engagement_score"] = engagement.clip(0, 1)
    
    # Add engagement bands
    df["engagement_band"] = pd.cut(
        df["engagement_score"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Moderate", "High"],
        include_lowest=True
    )
    
    print_success("Engagement scores calculated")
    print(f"  Mean engagement: {df['engagement_score'].mean():.3f}")
    print(f"  Distribution:")
    print(df["engagement_band"].value_counts().to_string().replace('\n', '\n  '))
    
    return df

def show_department_insights(df, dim_employee):
    """Display department-level engagement insights."""
    print_step("Department-level insights:")
    
    dept_summary = df.merge(
        dim_employee[["employee_id", "dept"]], on="employee_id"
    ).groupby("dept").agg({
        "engagement_score": "mean",
        "sentiment_compound": "mean",
        "q_workload": "mean",
        "q_recognition": "mean"
    }).round(3).sort_values("engagement_score", ascending=False)
    
    print(dept_summary.to_string())

# ============================================================================
# CLUSTERING
# ============================================================================

def cluster_employees(snapshot):
    """Segment employees into engagement cohorts using K-means."""
    print_step("Running K-means clustering...")
    
    feature_cols = ["q_workload", "q_alignment", "q_recognition", "q_growth", 
                    "sentiment_compound", "engagement_score"]
    X = snapshot[feature_cols].copy()
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal k using silhouette score
    print("  Evaluating cluster solutions...")
    best_k, best_score = None, -1
    for k in range(3, 6):
        kmeans = KMeans(n_clusters=k, random_state=Config.SEED, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"    k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score, best_k = score, k
    
    print_success(f"Optimal k={best_k} (silhouette={best_score:.3f})")
    
    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=Config.SEED, n_init="auto")
    snapshot["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Assign meaningful names based on engagement level
    cluster_means = snapshot.groupby("cluster")["engagement_score"].mean().sort_values()
    name_mapping = {
        cluster: name for cluster, name in zip(
            cluster_means.index,
            ["Disengaged", "Neutral", "Highly Engaged"][:len(cluster_means)]
        )
    }
    snapshot["cluster_name"] = snapshot["cluster"].map(name_mapping)
    
    # Summary
    print("\n  Cluster Distribution:")
    cluster_summary = snapshot.groupby("cluster_name").agg({
        "engagement_score": ["mean", "std"],
        "employee_id": "count"
    }).round(3)
    cluster_summary.columns = ["avg_engagement", "std_engagement", "n_employees"]
    print(cluster_summary.to_string().replace('\n', '\n  '))
    
    return snapshot

# ============================================================================
# TURNOVER RISK MODEL
# ============================================================================

def calculate_turnover_risk(snapshot, dim_employee):
    """Calculate interpretable turnover risk probability."""
    print_step("Calculating turnover risk...")
    
    # Calculate manager span (number of direct reports)
    mgr_span = dim_employee.groupby("manager_id")["employee_id"].count().rename("mgr_span")
    
    # Merge employee data
    risk_df = snapshot.merge(
        dim_employee[["employee_id", "tenure_months", "work_model", "manager_id", "dept"]],
        on="employee_id", how="left"
    ).merge(mgr_span, on="manager_id", how="left")
    
    # Normalize risk factors (0 = low risk, 1 = high risk)
    risk_df["engagement_risk"] = 1 - risk_df["engagement_score"]
    risk_df["tenure_risk"] = 1 - np.clip(risk_df["tenure_months"] / 36, 0, 1)
    risk_df["span_risk"] = np.clip((risk_df["mgr_span"] - 5) / 20, 0, 1)
    risk_df["remote_risk"] = (risk_df["work_model"] == "Remote").astype(int)
    
    # Weighted combination
    risk_raw = (
        Config.RISK_WEIGHTS["engagement"] * risk_df["engagement_risk"] +
        Config.RISK_WEIGHTS["tenure"] * risk_df["tenure_risk"] +
        Config.RISK_WEIGHTS["span"] * risk_df["span_risk"] +
        Config.RISK_WEIGHTS["remote"] * risk_df["remote_risk"]
    )
    risk_df["turnover_risk_prob"] = np.clip(risk_raw, 0, 1)
    
    # Risk categories
    risk_df["risk_category"] = pd.cut(
        risk_df["turnover_risk_prob"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    
    # Summary
    print_success("Risk scores calculated")
    high_risk_count = (risk_df["turnover_risk_prob"] > 0.7).sum()
    high_risk_pct = high_risk_count / len(risk_df) * 100
    print(f"  Mean risk: {risk_df['turnover_risk_prob'].mean():.3f}")
    print(f"  High risk (>0.7): {high_risk_count} employees ({high_risk_pct:.1f}%)")
    print("\n  Risk Distribution:")
    print(risk_df["risk_category"].value_counts().to_string().replace('\n', '\n  '))
    
    return risk_df

def analyze_high_risk_cohorts(risk_df):
    """Identify and analyze high-risk employee cohorts."""
    print_step("High-risk cohort analysis:")
    
    high_risk = risk_df[risk_df["turnover_risk_prob"] >= 0.7]
    
    # By department
    print("\n  Top Departments at Risk:")
    dept_risk = high_risk.groupby("dept").agg({
        "employee_id": "count",
        "turnover_risk_prob": "mean"
    }).sort_values("employee_id", ascending=False).head(5)
    dept_risk.columns = ["n_high_risk", "avg_risk"]
    print(dept_risk.to_string().replace('\n', '\n  '))
    
    # By manager
    print("\n  Top Managers with High-Risk Reports:")
    mgr_risk = high_risk.groupby("manager_id").agg({
        "employee_id": "count",
        "turnover_risk_prob": "mean",
        "engagement_score": "mean"
    }).sort_values("employee_id", ascending=False).head(5)
    mgr_risk.columns = ["n_high_risk", "avg_risk", "avg_engagement"]
    print(mgr_risk.to_string().replace('\n', '\n  '))

# ============================================================================
# EXPORT & INSIGHTS
# ============================================================================

def export_data(dim_employee, fact_pulse, risk_df):
    """Export data to CSV files for Power BI."""
    print_section("EXPORTING DATA FOR POWER BI")
    
    ensure_data_dir()
    
    # Save main tables
    dim_employee.to_csv("data/dim_employee.csv", index=False)
    fact_pulse.to_csv("data/fact_pulse_enriched.csv", index=False)
    
    # Save features snapshot
    features_output = risk_df[[
        "employee_id", "engagement_score", "cluster", "cluster_name",
        "turnover_risk_prob", "risk_category"
    ]]
    features_output.to_csv("data/features_latest.csv", index=False)
    
    print("\n✓ Exported files:")
    print(f"  • data/dim_employee.csv ({len(dim_employee)} rows)")
    print(f"  • data/fact_pulse_enriched.csv ({len(fact_pulse)} rows)")
    print(f"  • data/features_latest.csv ({len(features_output)} rows)")

def show_key_insights(fact_pulse, risk_df, dim_employee):
    """Display summary insights and recommendations."""
    print_section("KEY INSIGHTS & RECOMMENDATIONS")
    
    # Top keywords
    print_step("Top Keywords in Comments:")
    words = []
    for comment in fact_pulse["comment_text"].dropna():
        tokens = re.findall(r'\b[a-z]+\b', comment.lower())
        stopwords = {'and', 'the', 'is', 'are', 'for', 'of', 'to', 'in', 'a', 'my', 'have', 'has', 'but', 'with'}
        words.extend([w for w in tokens if w not in stopwords and len(w) > 4])
    
    top_words = Counter(words).most_common(10)
    for word, count in top_words:
        print(f"  • {word}: {count}")
    
    # Key metrics
    high_risk = risk_df[risk_df["turnover_risk_prob"] >= 0.7]
    highly_engaged = risk_df[risk_df["cluster_name"] == "Highly Engaged"]
    
    print_step("Summary Metrics:")
    print(f"  • Total Employees: {len(dim_employee)}")
    print(f"  • Average Engagement: {fact_pulse['engagement_score'].mean():.3f}")
    print(f"  • Highly Engaged: {len(highly_engaged)} ({len(highly_engaged)/len(risk_df)*100:.1f}%)")
    print(f"  • At Risk Employees: {len(high_risk)} ({len(high_risk)/len(risk_df)*100:.1f}%)")
    
    # Top risk departments
    dept_risk = high_risk.groupby("dept")["employee_id"].count().sort_values(ascending=False)
    if len(dept_risk) > 0:
        print(f"  • Departments Needing Attention: {', '.join(dept_risk.head(3).index.tolist())}")
    
    print_step("Recommended Actions:")
    print("  1. Focus retention efforts on Support and Operations teams")
    print("  2. Review workload distribution for remote workers")
    print("  3. Conduct 1-on-1s with managers who have >3 high-risk reports")
    print("  4. Implement recognition programs in low-scoring departments")
    print("  5. Monitor disengaged cluster monthly for early warning signs")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete analytics pipeline."""
    print_section("EMPLOYEE ENGAGEMENT ANALYTICS PIPELINE")
    print("\nStarting complete data generation and analysis...")
    
    # Phase 1: Data Generation
    print_section("PHASE 1: GENERATING SYNTHETIC DATA")
    dim_employee = generate_employees()
    fact_pulse = generate_pulse_surveys(dim_employee)
    
    # Phase 2: NLP Analysis
    print_section("PHASE 2: NLP SENTIMENT ANALYSIS")
    fact_pulse = analyze_sentiment(fact_pulse)
    
    # Phase 3: Engagement Scoring
    print_section("PHASE 3: CALCULATING ENGAGEMENT SCORES")
    fact_pulse = calculate_engagement_scores(fact_pulse)
    show_department_insights(fact_pulse, dim_employee)
    
    # Phase 4: Clustering & Risk
    print_section("PHASE 4: CLUSTERING & TURNOVER RISK")
    latest_month = fact_pulse["month"].max()
    snapshot = fact_pulse[fact_pulse["month"] == latest_month].copy()
    print(f"\n→ Analyzing snapshot: {latest_month} ({len(snapshot)} employees)")
    
    snapshot = cluster_employees(snapshot)
    risk_df = calculate_turnover_risk(snapshot, dim_employee)
    analyze_high_risk_cohorts(risk_df)
    
    # Export
    export_data(dim_employee, fact_pulse, risk_df)
    
    # Insights
    show_key_insights(fact_pulse, risk_df, dim_employee)
    
    # Completion
    print_section("✓ PIPELINE COMPLETE!")
    print("\nNext Steps:")
    print("  1. Open Power BI Desktop")
    print("  2. Import the 3 CSV files from /data folder")
    print("  3. Create relationships:")
    print("     - dim_employee (1) ←→ (∞) fact_pulse_enriched [employee_id]")
    print("     - dim_employee (1) ←→ (1) features_latest [employee_id]")
    print("  4. Build dashboard with KPIs, trends, and risk analysis")
    print("  5. Review insights and prepare presentation")
    print("\n" + "=" * 80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        print("\nPlease ensure all required packages are installed:")
        print("  pip install pandas numpy faker python-dateutil vaderSentiment scikit-learn matplotlib seaborn")
        sys.exit(1)