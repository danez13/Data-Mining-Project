import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

module = __import__("generate-rules11")
Apriori = getattr(module, "Apriori")

# rules_file = "rules11.txt"

def load_info(info_file="output_files/info11.txt"):
    if not os.path.exists(info_file):
        print("Error: Required files are missing. Ensure that the Apriori algorithm has been run successfully.")
        return None
    # Load info11.txt (Summary Data)
    summary_data = {}
    with open(info_file, "r") as file:
        for line in file:
            if ":" in line:
                key, value = line.strip().split(": ", 1)
                summary_data[key.strip()] = value.strip()
    return summary_data

# plot 1: generation time
def plot_time(times:pd.DataFrame):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=times, x="support", y="time", palette="viridis")
    plt.xlabel("Support")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time for Apriori Algorithm")
    plt.savefig("visualizations/execution_time.png")
    plt.show()

# plot 2: frequent itemsets count
def plot_freq_count(counts:pd.DataFrame):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=counts, x="support", y="count", palette="viridis")
    plt.xlabel("Support")
    plt.ylabel("Number of Frequent Itemsets")
    plt.title("Number of Frequent Itemsets for Different Support Thresholds")
    plt.savefig("visualizations/frequent_itemsets_count.png")
    plt.show()

# plot 3: high-confidence rules count
def plot_rule_count(counts:pd.DataFrame):
    plt.figure(figsize=(8, 4))
    plt.bar(counts["confidence"], counts["count"])
    plt.xlabel("Confidence")
    plt.ylabel("Number of High-Confidence Rules")
    plt.title("Number of High-Confidence Rules for Different Confidence Thresholds")
    plt.savefig("visualizations/high_confidence_rules_count.png")
    plt.show()

input_file="small.txt"

"""data collection"""
# question 2 & 3
times = {"support":[],"time":[]}
freq_counts = {"support":[],"count":[]}
for supp in [50,75,100,125,150,200]:
    print(f"processing Support: {supp}...")
    algo = Apriori(supp,0.8,input_file)
    algo.run()
    print("data generated.")
    print(f"loading info data...")
    info = load_info()
    assert info
    print(f"info data loaded.")
    print("retrieving data...")
    times["support"].append(supp)
    freq_counts["support"].append(supp)
    times["time"].append(float(info["Time in seconds to find the frequent itemsets"]))
    freq_counts["count"].append(float(info["Total number of frequent itemsets"]))
    print(f"data retrieved.\n")

# question 4
rule_counts = {"confidence":[],"count":[]}
for conf in [0.9,0.85,0.8,0.75,0.7]:
    print(f" processing Confidence: {conf}...")
    algo = Apriori(100,conf,input_file)
    algo.run()
    print("data generated.")
    print(f"loading info data...")
    info = load_info()
    assert info
    print(f"info data loaded.")
    print("retrieving data...")
    count = info["Number of high-confidence rules"]
    rule_counts["confidence"].append(str(conf))
    rule_counts["count"].append(count)
    print(f"data retrieved.")

"""plotting"""
os.makedirs("visualizations",exist_ok=True)
# question 2
time_df = pd.DataFrame(times)
print(time_df)
plot_time(time_df)

# question 3
count_df = pd.DataFrame(freq_counts)
print(count_df)
plot_freq_count(count_df)

# question 4
rule_df = pd.DataFrame(rule_counts)
print(rule_df)
plot_rule_count(rule_df)