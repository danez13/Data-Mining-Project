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

# Plot 1: Bar Chart of Frequent Itemsets vs. Support Count with Improved Data Labels
plt.figure(figsize=(8, 6))
sns.barplot(x=items_df["Itemset Size"], y=items_df["Support Count"], palette="Blues")
plt.xlabel("Itemset Size")
plt.ylabel("Support Count")
plt.title("Frequent Itemsets Support Count")
plt.xticks(rotation=45)

# Add data labels to bar chart
for i, v in enumerate(items_df["Support Count"]):
    plt.text(i, v + 100, f'{v} ({items_df["Itemset Size"][i]})', ha='center', fontsize=10)

plt.show()

# Plot 2: Histogram of Frequent Itemset Report
plt.figure(figsize=(7, 5))
plt.hist(items_df["Support"], bins=10, color="blue", alpha=0.7)
plt.xlabel("Support Value")
plt.ylabel("Frequency")
plt.title("Distribution of Support for Frequent Itemsets")
plt.show()

# Plot 3: Pie Chart of Frequent Itemset Distribution (if multiple itemsets exist)
if len(items_df) > 1:
    plt.figure(figsize=(7, 7))
    plt.pie(items_df["Support"], labels=items_df["Itemset Size"], autopct='%1.1f%%', colors=sns.color_palette("Blues"))
    plt.title("Distribution of Frequent Itemsets")
    plt.show()

# Plot 4: Line Plot of Execution Time
plt.figure(figsize=(8, 4))
plt.plot(["Frequent Itemsets", "Confident Rules"], [exec_time_itemsets, exec_time_rules], marker='o', linestyle='-')
plt.xlabel("Process")
plt.ylabel("Time (seconds)")
plt.title("Execution Time for Apriori Algorithm")
plt.show()

# Plot 5: Scatter plot of Itemset Size vs. Support Count
plt.figure(figsize=(8, 6))
plt.scatter(items_df["Itemset Size"], items_df["Support Count"], color="blue", alpha=0.7)
plt.xlabel("Itemset Size")
plt.ylabel("Support Count")
plt.title("Itemset Size vs. Support Count")
plt.show()

# Check if rules11.txt has data before attempting to plot
if os.path.exists(rules_file) and os.path.getsize(rules_file) > 0:
    print("Rules found! Additional visualizations can be added if needed.")
else:
    print("No association rules were generated. Try lowering the min confidence or support thresholds.")
