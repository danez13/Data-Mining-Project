import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Importing the Apriori class from the module that generates rules
module = __import__("generate-rules11")
Apriori = getattr(module, "Apriori")

def load_info(info_file="output_files/info11.txt"):
    """
    Load the summary information generated by the Apriori algorithm from the info file.
    
    Parameters:
    - info_file (str): Path to the information summary file generated by the Apriori algorithm.
    
    Returns:
    - dict: A dictionary containing key-value pairs from the info file.
    - None: If the file doesn't exist or can't be read, prints an error message and returns None.
    """
    if not os.path.exists(info_file):
        print("Error: Required files are missing. Ensure that the Apriori algorithm has been run successfully.")
        return None
    
    # Initialize a dictionary to store the summary data
    summary_data = {}
    with open(info_file, "r") as file:
        for line in file:
            if ":" in line:
                # Split the line by the first occurrence of ': ' and store key-value pairs in the dictionary
                key, value = line.strip().split(": ", 1)
                summary_data[key.strip()] = value.strip()
    
    return summary_data

# Plot 1: Generation time vs support threshold
def plot_time(times: pd.DataFrame):
    """
    Generate and save a bar plot showing the execution time of the Apriori algorithm for different support thresholds.
    
    Parameters:
    - times (pd.DataFrame): DataFrame containing the support thresholds and corresponding execution times.
    
    Returns:
    - None: Displays the plot and saves it as 'execution_time.png'.
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(data=times, x="support", y="time", palette="viridis")
    plt.xlabel("Support")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time for Apriori Algorithm")
    plt.savefig("visualizations/execution_time.png")
    plt.show()

# Plot 2: Frequent itemsets count vs support threshold
def plot_freq_count(counts: pd.DataFrame):
    """
    Generate and save a bar plot showing the number of frequent itemsets for different support thresholds.
    
    Parameters:
    - counts (pd.DataFrame): DataFrame containing the support thresholds and corresponding frequent itemset counts.
    
    Returns:
    - None: Displays the plot and saves it as 'frequent_itemsets_count.png'.
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(data=counts, x="support", y="count", palette="viridis")
    plt.xlabel("Support")
    plt.ylabel("Number of Frequent Itemsets")
    plt.title("Number of Frequent Itemsets for Different Support Thresholds")
    plt.savefig("visualizations/frequent_itemsets_count.png")
    plt.show()

# Plot 3: High-confidence rules count vs confidence threshold
def plot_rule_count(counts: pd.DataFrame):
    """
    Generate and save a bar plot showing the number of high-confidence rules for different confidence thresholds.
    
    Parameters:
    - counts (pd.DataFrame): DataFrame containing the confidence thresholds and corresponding high-confidence rule counts.
    
    Returns:
    - None: Displays the plot and saves it as 'high_confidence_rules_count.png'.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(counts["confidence"], counts["count"])
    plt.xlabel("Confidence")
    plt.ylabel("Number of High-Confidence Rules")
    plt.title("Number of High-Confidence Rules for Different Confidence Thresholds")
    plt.savefig("visualizations/high_confidence_rules_count.png")
    plt.show()

# Input file for Apriori algorithm
input_file = "small.txt"

"""Data Collection for Analysis"""

# Question 2 & 3: Collecting data for different support thresholds
# times will store execution times for each support threshold
# freq_counts will store the number of frequent itemsets found for each support threshold
times = {"support": [], "time": []}
freq_counts = {"support": [], "count": []}

# Iterate through different support thresholds (50, 75, 100, 125, 150, 200)
for supp in [50, 75, 100, 125, 150, 200]:
    print(f"Processing Support: {supp}...")  # Display current support threshold
    algo = Apriori(supp, 0.75, input_file)  # Initialize the Apriori algorithm with the current support
    algo.run()  # Run the Apriori algorithm
    print("Data generated.")
    
    print(f"Loading info data...")  # Load summary info after running the algorithm
    info = load_info()  # Load the summary info from the 'info11.txt' file
    assert info  # Ensure that info is successfully loaded
    print(f"Info data loaded.")
    
    print("Retrieving data...")
    # Append the collected data (execution time and number of frequent itemsets)
    times["support"].append(supp)
    freq_counts["support"].append(supp)
    times["time"].append(float(info["Time in seconds to find the frequent itemsets"]))
    freq_counts["count"].append(float(info["Total number of frequent itemsets"]))
    print(f"Data retrieved.\n")

# Question 4: Collecting data for different confidence thresholds
# rule_counts will store the count of high-confidence rules for each confidence threshold
rule_counts = {"confidence": [], "count": []}

# Iterate through different confidence thresholds (0.9, 0.85, 0.8, 0.75, 0.7)
for conf in [0.9, 0.85, 0.8, 0.75, 0.7]:
    print(f"Processing Confidence: {conf}...")  # Display current confidence threshold
    algo = Apriori(100, conf, input_file)  # Initialize the Apriori algorithm with the current confidence
    algo.run()  # Run the Apriori algorithm
    print("Data generated.")
    
    print(f"Loading info data...")  # Load summary info after running the algorithm
    info = load_info()  # Load the summary info from the 'info11.txt' file
    assert info  # Ensure that info is successfully loaded
    print(f"Info data loaded.")
    
    print("Retrieving data...")
    # Append the collected data (confidence threshold and number of high-confidence rules)
    count = info["Number of high-confidence rules"]
    rule_counts["confidence"].append(str(conf))  # Convert confidence to string for compatibility with plotting
    rule_counts["count"].append(count)  # Append the count of high-confidence rules
    print(f"Data retrieved.")

"""Plotting Results"""

# Create directory for saving visualizations if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# Question 2: Plot execution time for different support thresholds
time_df = pd.DataFrame(times)  # Create a DataFrame from the collected times data
print(time_df)
plot_time(time_df)  # Call the plotting function for execution times

# Question 3: Plot the number of frequent itemsets for different support thresholds
count_df = pd.DataFrame(freq_counts)  # Create a DataFrame from the collected frequent itemset counts
print(count_df)
plot_freq_count(count_df)  # Call the plotting function for frequent itemset counts

# Question 4: Plot the number of high-confidence rules for different confidence thresholds
rule_df = pd.DataFrame(rule_counts)  # Create a DataFrame from the collected rule counts data
print(rule_df)
plot_rule_count(rule_df)  # Call the plotting function for high-confidence rule counts
