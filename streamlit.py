import streamlit as st
import json
import os
import re  # For extracting numbers safely
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load and format model evaluation results from a text file."""
    file_path = "model_evaluation_results.txt"

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = f.readlines()
                structured_data = parse_text_results(data)
                return structured_data
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return {}
    else:
        st.warning("Warning: The results file does not exist.")
        return {}

def extract_class_distribution(text):
    """Extract class distribution data safely using regex."""
    match = re.findall(r"(\d+): (\d+)", text)  # Find all number pairs
    return {int(k): int(v) for k, v in match} if match else {}

def parse_text_results(lines):
    """Parse raw text results and convert them into structured JSON format."""
    results = {}
    current_section = None
    current_data = []

    for line in lines:
        line = line.strip()

        if line.startswith("Classification Report:"):
            current_section = "classification_report"
            current_data = []

        elif line.startswith("Class Distribution:"):
            if current_section == "classification_report":
                results[current_section] = "\n".join(current_data)  
            current_section = "class_distribution"
            current_data = []

        elif line.startswith("Train Set:"):
            results["train_set"] = extract_class_distribution(line)

        elif line.startswith("Test Set:"):
            results["test_set"] = extract_class_distribution(line)

        elif line.startswith("SMOTE Train Set:"):
            results["smote_train_set"] = extract_class_distribution(line)

        elif current_section:
            current_data.append(line)

    if current_section and current_data:
        results[current_section] = "\n".join(current_data)

    return results

def plot_class_distribution(results):
    """Visualize class distribution in Train, Test, and SMOTE datasets."""
    if "train_set" not in results or "test_set" not in results or "smote_train_set" not in results:
        st.warning("Class distribution data missing.")
        return
    
    data = []
    for dataset, values in results.items():
        if dataset in ["train_set", "test_set", "smote_train_set"]:
            for class_label, count in values.items():
                data.append({"Dataset": dataset.replace("_", " ").title(), "Class": class_label, "Count": count})
    
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Dataset", y="Count", hue="Class", palette="viridis", ax=ax)
    ax.set_title("Class Distribution Across Datasets")
    ax.set_xlabel("Dataset Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_confusion_matrix():
    """Mock confusion matrix for visualization (Replace with actual data if available)."""
    cm_data = [[1, 0], [0, 1]]  # Replace with actual values
    labels = ["Class 0", "Class 1"]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

def main():
    st.title("🚀 Threat Detection Model Results")

    results = load_results()

    if results:
        # Display Classification Report
        st.subheader("📊 Classification Report")
        st.text(results.get("classification_report", "N/A"))

        # Show Class Distribution
        st.subheader("📉 Class Distribution")
        plot_class_distribution(results)

        # Show Confusion Matrix
        st.subheader("🧩 Confusion Matrix")
        plot_confusion_matrix()

        # JSON Download
        json_results = json.dumps(results, indent=4) if results else "{}"
        st.sidebar.download_button(
            "⬇️ Download JSON Report", json_results, "model_results.json", "application/json"
        )
    else:
        st.warning("No results found or invalid file format.")

if __name__ == "__main__":
    main()

