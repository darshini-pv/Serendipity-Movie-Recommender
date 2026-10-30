
import os
import pandas as pd

# ============================================================
# STAGE 19: PROJECT AUDIT
# ============================================================

PROJECT_DIR = (
    r"C:\Users\manoj\OneDrive\Desktop"
    r"\Serendipitous Recommendation Engine"
)

print("=" * 70)
print("STAGE 19: PROJECT AUDIT")
print("=" * 70)

# Important output files from Stages 14 to 18
expected_files = [
    "stage14_validation_tuning_results.csv",
    "stage14_final_test_recommendation_results.csv",
    "stage14_final_test_rating_accuracy.csv",
    "stage14_final_model_summary.csv",
    "stage16_model_comparison.csv",
    "stage16_user_level_metrics.csv",
    "stage16_paired_comparison_summary.csv",
    "stage17_serendipity_summary.csv",
    "stage17_user_level_metrics.csv",
    "stage17_recommendation_details.csv",
    "stage18_representative_recommendations.csv",
    "stage18_relevant_vs_nonrelevant.csv",
    "stage18_rank_analysis.csv",
    "stage18_genre_analysis.csv"
]

print("\nChecking project output files...\n")

audit_results = []

for filename in expected_files:

    filepath = os.path.join(PROJECT_DIR, filename)

    if os.path.exists(filepath):

        try:
            df = pd.read_csv(filepath)

            status = "FOUND"

            print(f"[FOUND] {filename}")
            print(f"        Rows: {df.shape[0]}")
            print(f"        Columns: {df.shape[1]}")

        except Exception as error:

            status = "READ ERROR"

            print(f"[READ ERROR] {filename}")
            print(f"             {error}")

    else:

        status = "MISSING"

        print(f"[MISSING] {filename}")

    audit_results.append({
        "File": filename,
        "Status": status
    })

# Save audit report
audit_df = pd.DataFrame(audit_results)

output_path = os.path.join(
    PROJECT_DIR,
    "stage19_project_audit.csv"
)

audit_df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)

print("\nAudit report saved to:")
print(output_path)

print("\nFile status summary:")
print(audit_df["Status"].value_counts())
