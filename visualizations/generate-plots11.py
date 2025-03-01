import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check if files exist
items_file = "items11.txt"
info_file = "info11.txt"
rules_file = "rules11.txt"

if not os.path.exists(items_file) or not os.path.exists(info_file):
    print("Error: Required files are missing. Ensure that the Apriori algorithm has been run successfully.")
    exit()

# Load items11.txt (Frequent Itemsets)
frequent_itemsets = []
with open(items_file, "r") as file:
    for line in file:
        parts = line.strip().split("|")
        if len(parts) == 3:
            itemset_size, support_count, support = parts
            frequent_itemsets.append([int(itemset_size), int(support_count), float(support)])

# Convert to DataFrame
items_df = pd.DataFrame(frequent_itemsets, columns=["Itemset Size", "Support Count", "Support"])

# Load info11.txt (Summary Data)
summary_data = {}
with open(info_file, "r") as file:
    for line in file:
        if ":" in line:
            key, value = line.strip().split(": ", 1)
            summary_data[key.strip()] = value.strip()

# Convert execution times to float
exec_time_itemsets = float(summary_data.get("Time in seconds to find the frequent itemsets", 0))
exec_time_rules = float(summary_data.get("Time in seconds to find the confident rules", 0))

## Plot 1: Bar Chart of Frequent Itemsets vs. Support Count ###
plt.figure(figsize=(8, 6))
sns.barplot(x=items_df["Itemset Size"], y=items_df["Support Count"], palette="Blues")
plt.xlabel("Itemset Size")
plt.ylabel("Support Count")
plt.title("Frequent Itemsets Support Count")
plt.xticks(rotation=45)

# Add data labels to bar chart
for i, v in enumerate(items_df["Support Count"]):
    plt.text(i, v + 50, str(v), ha='center', fontsize=10)

plt.savefig("visualizations/frequent_itemsets_support.png")
plt.show()

## Plot 2: Improved Bar Chart for Support Values ###
plt.figure(figsize=(6, 4))
plt.bar(items_df["Itemset Size"].astype(str), items_df["Support"], color="skyblue", edgecolor="black")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.title("Support Value of Frequent Itemsets")

# Add Data Labels for Support Values
for i, v in enumerate(items_df["Support"]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.savefig("visualizations/frequent_itemsets_support_values_fixed.png")
plt.show()

# Conditional Pie Chart (Only If Multiple Itemsets Exist) ###
if len(items_df) > 1:
    plt.figure(figsize=(7, 7))
    plt.pie(items_df["Support"], labels=items_df["Itemset Size"], autopct='%1.1f%%', colors=sns.color_palette("Blues"))
    plt.title("Distribution of Frequent Itemsets")
    plt.savefig("visualizations/frequent_itemsets_pie.png")
    plt.show()
else:
    print("Skipping pie chart: Only one frequent itemset found.")

# Plot 3: Execution Time Bar Chart (Hides Empty Bar) ###
processes = ["Frequent Itemsets"]
times = [exec_time_itemsets]
if exec_time_rules > 0:
    processes.append("Confident Rules")
    times.append(exec_time_rules)

plt.figure(figsize=(8, 4))
plt.bar(processes, times, color=["blue", "red"])
plt.xlabel("Process")
plt.ylabel("Time (seconds)")
plt.title("Execution Time for Apriori Algorithm")
plt.savefig("visualizations/execution_time.png")
plt.show()

# Plot 4: Box Plot for Support Count Distribution (Fix for Single Itemset) ###
plt.figure(figsize=(7, 5))
if len(items_df) > 1:
    sns.boxplot(y=items_df["Support Count"], color="blue")
else:
    plt.scatter(1, items_df["Support Count"].values[0], color="blue", label="Single Itemset")
    plt.legend()

plt.ylabel("Support Count")
plt.title("Box Plot of Frequent Itemset Support Counts")
plt.savefig("visualizations/support_count_distribution.png")
plt.show()

#  Check if rules11.txt has data before attempting to plot association rules
if os.path.exists(rules_file) and os.path.getsize(rules_file) > 0:
    print("Rules found! Additional visualizations can be added if needed.")
else:
    print("No association rules were generated. Try lowering the min confidence or support thresholds.")
