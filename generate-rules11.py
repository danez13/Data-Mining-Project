from itertools import combinations
import time
import os

class Apriori:
    def __init__(self, min_support, min_confidence, file_path):
        """
        Initialize the Apriori algorithm with minimum support and confidence thresholds.
        
        Parameters:
        min_support (float): The minimum support threshold to consider itemsets frequent.
        min_confidence (float): The minimum confidence threshold for generating association rules.
        file_path (string): Path to the file containing transaction data.

        Output:
        None
        """
        self.min_support = min_support  # Minimum support threshold for itemsets
        self.min_confidence = min_confidence  # Minimum confidence threshold for rules
        self.frequent_itemsets = {}  # Dictionary to store frequent itemsets with their support counts, organized by size
        self.rules = []  # List to store generated association rules
        self.file_path = file_path  # Path to the input file containing transactions
        self.transactions = []  # List to hold transactions (set of items per transaction)
        self.parse_file()  # Parse the input file and populate the transactions list
        self.num_transactions = len(self.transactions)  # Number of transactions in the dataset

    def generate_candidates(self, prev_frequent_itemsets, k):
        """
        Generate candidate itemsets of size k from previous frequent itemsets.

        Parameters:
        prev_frequent_itemsets (set of frozensets): Frequent itemsets from the previous iteration.
        k (int): The size of itemsets to generate.

        Output:
        set of frozensets: Generated candidate itemsets of size k+1.
        """                                                 
        candidates = set()
        for itemset in prev_frequent_itemsets:
            for compare_itemset in prev_frequent_itemsets:
                item = itemset.union(compare_itemset)
                if len(item) == k+1:  # Only consider itemsets of size k+1
                    candidates.add(frozenset(item))
        return candidates
    
    def prune_candidates(self, candidates, prev_frequent_itemsets):
        """
        Prune candidate itemsets by removing those that contain non-frequent subsets.

        Parameters:
        candidates (set of frozensets): Candidate itemsets to be pruned.
        prev_frequent_itemsets (set of frozensets): Previous frequent itemsets used to check subsets.

        Output:
        set of frozensets: Pruned candidate itemsets.
        """
        pruned = set()
        for candidate in candidates:
            # Check all (k-1)-subsets of the candidate and ensure they are frequent
            is_valid = True
            for subset in combinations(candidate, len(candidate) - 1):
                if frozenset(subset) not in prev_frequent_itemsets:
                    is_valid = False
                    break
            
            if is_valid:
                pruned.add(candidate)
        
        return pruned
    
    def count_support(self, candidates):
        """
        Count the support of each candidate itemset in the transactions.

        Parameters:
        candidates (set of frozensets): Candidate itemsets whose support needs to be counted.

        Output:
        dict: Dictionary mapping candidate itemsets to their support counts.
        """
        support_counts = {}
        for candidate in candidates:
            support_count = 0
            for transaction in self.transactions:
                if candidate.issubset(transaction):  # Check if candidate is a subset of transaction
                    support_count += 1
            support_counts[candidate] = support_count
        return support_counts
    
    def eliminate_infrequent(self, candidates):
        """
        Eliminate candidates that do not meet the minimum support threshold.

        Parameters:
        candidates (dict): Dictionary mapping itemsets to their support counts.

        Output:
        dict: Dictionary of itemsets that meet the support threshold.
        """
        frequent_itemsets = {}
        for candidate, support_count in candidates.items():
            if support_count < self.min_support:  # Eliminate itemsets that don't meet min support
                continue
            frequent_itemsets[candidate] = support_count
        return frequent_itemsets
    
    def find_frequent_itemsets(self):
        """
        Iteratively find frequent itemsets using the Apriori algorithm steps.
        
        This method applies the main steps of the Apriori algorithm to find frequent itemsets:
        1. Generate candidate itemsets.
        2. Count their support.
        3. Prune itemsets that are infrequent.
        4. Continue until no more frequent itemsets are found.

        Output:
        None (Updates self.frequent_itemsets dictionary).
        """
        k = 1
        # Generate candidate 1-itemsets from transactions
        current_candidates_itemsets = {frozenset([item]) for transaction in self.transactions for item in transaction}
        support_counts = self.count_support(current_candidates_itemsets)
        
        self.frequent_itemsets = {}  # Initialize dictionary to store frequent itemsets
        self.frequent_itemsets[1] = self.eliminate_infrequent(support_counts)
        current_frequent_itemsets = self.frequent_itemsets[1]
        frequent_itemsets_k = self.frequent_itemsets[1]
        
        while current_frequent_itemsets:
            # Generate candidate itemsets of size k+1
            candidates = self.generate_candidates(set(frequent_itemsets_k.keys()), k)

            # Prune candidates by eliminating those that contain non-frequent subsets
            current_candidates_itemsets = self.prune_candidates(candidates, set(frequent_itemsets_k.keys()))

            # Count support for current candidates
            support_counts = self.count_support(current_candidates_itemsets)

            # Eliminate infrequent itemsets
            frequent_itemsets_k = self.eliminate_infrequent(support_counts)

            # If no frequent itemsets found, stop the loop
            if not frequent_itemsets_k:
                break

            # Increment k to explore larger itemsets
            k += 1

            # Store frequent itemsets for this size category
            self.frequent_itemsets[k] = frequent_itemsets_k

            current_frequent_itemsets = frequent_itemsets_k
    
    def generate_item_output(self):
        """
        Generate the output file for frequent itemsets with their support.

        Output:
        None (Writes to file "output_files/items11.txt").
        """
        os.makedirs("output_files", exist_ok=True)  # Ensure the output directory exists
        with open("output_files/items11.txt", "w") as file:
            for itemsets in self.frequent_itemsets.values():
                for itemset, support in itemsets.items():
                    file.write(f"{' '.join(map(str, itemset))}|{support}|{support/self.num_transactions}\n")

    def generate_rule_output(self):
        """
        Generate the output file for association rules with their metrics.

        Output:
        None (Writes to file "output_files/rules11.txt").
        """
        os.makedirs("output_files", exist_ok=True)  # Ensure output directory exists
        with open("output_files/rules11.txt", "w") as file:
            for rule in self.rules:
                itemset = frozenset(rule[0].split(",") + rule[1].split(","))
                con = frozenset(rule[1].split(","))
                support_count = self.frequent_itemsets[len(itemset)][itemset]
                con_support = self.frequent_itemsets[len(con)][con] / self.num_transactions
                rule_confidence = rule[2]
                lift = rule_confidence / con_support

                antecedent = " ".join(map(str, rule[0].split(',')))
                consequent = " ".join(map(str, rule[1].split(',')))
                
                file.write(f"{antecedent}|{consequent}|{support_count}|{support_count/self.num_transactions}|{rule_confidence}|{lift}\n")

    def generate_info_output(self, find_frequent_time, rule_time):
        """
        Generate the information output file with stats and timings.

        Output:
        None (Writes to file "output_files/info11.txt").
        """
        os.makedirs("output_files", exist_ok=True)  # Ensure output directory exists
        with open("output_files/info11.txt", "w") as file:
            file.write(f"minsuppc: {self.min_support}\n")
            file.write(f"minconf: {self.min_confidence}\n")
            file.write(f"input file: {self.file_path}\n")
            file.write(f"Number of items: {len(set().union(*self.transactions))}\n")
            file.write(f"Number of transactions: {self.num_transactions}\n")
            file.write(f"The length of the longest transaction: {max(map(len, self.transactions))}\n")
            total_frequent_itemset_count = 0
            for k in self.frequent_itemsets:
                k_frequent_itemset_count = len(self.frequent_itemsets[k])
                if k_frequent_itemset_count == 0:
                    break
                file.write(f"Number of frequent {k}-itemsets: {k_frequent_itemset_count}\n")
                total_frequent_itemset_count += k_frequent_itemset_count
            file.write(f"Total number of frequent itemsets: {total_frequent_itemset_count}\n")
            file.write(f"Number of high-confidence rules: {len(self.rules)}\n")
            
            # Finding rule with highest confidence and lift
            confident_rule = lifted_rule = None
            best_lift = float("-inf")
            if len(self.rules) != 0:
                for rule in self.rules:
                    con = frozenset(rule[1].split(","))
                    con_support = self.frequent_itemsets[len(con)][con] / self.num_transactions
                    lift = rule[2] / con_support
                    if confident_rule is None or confident_rule[2] < rule[2]:
                        confident_rule = rule
                    if lifted_rule is None or best_lift < lift:
                        lifted_rule = rule
                        best_lift = lift
                assert confident_rule
                assert lifted_rule

            file.write(f"The rule with the highest confidence: {confident_rule}\n")
            file.write(f"The rule with the highest lift: {lifted_rule}\n")
            file.write(f"Time in seconds to find the frequent itemsets: {find_frequent_time}\n")
            if rule_time > 0:
                file.write(f"Time in seconds to find the confident rules: {rule_time}\n")
    
    def generate_rules(self):
        """
        Generate association rules from the discovered frequent itemsets.

        Parameters:
        None (Uses self.frequent_itemsets)

        Output:
        None (Updates self.rules)
        """
        for k, itemsets in self.frequent_itemsets.items():
            if k == 1:  # Skip 1-itemsets because they can't generate association rules
                continue
            
            for itemset, support in itemsets.items():
                # Generate all non-empty subsets of the itemset
                subsets = list(combinations(itemset, len(itemset)-1))
                for subset in subsets:
                    antecedent = frozenset(subset)
                    consequent = itemset - antecedent  # The rest of the itemset is the consequent
                    
                    # Get the support of the antecedent and consequent
                    antecedent_support = self.frequent_itemsets[len(antecedent)].get(antecedent, 0)
                    
                    if antecedent_support > 0:  # Confidence can be calculated only if antecedent support > 0
                        confidence = support / antecedent_support  # Calculate confidence

                        # If confidence meets the minimum threshold, add the rule
                        if confidence >= self.min_confidence:
                            rule = (",".join(antecedent), ",".join(consequent), confidence)
                            self.rules.append(rule)

    def run(self):
        """
        Execute the Apriori algorithm: find frequent itemsets and generate rules.

        Output:
        tuple: (frequent_itemsets, rules)
        """
        start = time.time()
        self.find_frequent_itemsets()  # Step 1: Find frequent itemsets
        end = time.time()
        find_frequent_time = end - start
        self.generate_item_output()  # Step 2: Generate itemset output

        if self.min_confidence != -1:
            start = time.time()
            self.generate_rules()  # Step 3: Generate rules based on frequent itemsets
            end = time.time()
            rule_time = end - start
            self.generate_rule_output()  # Step 4: Generate rules output
        else:
            rule_time = -1
        self.generate_info_output(find_frequent_time, rule_time)  # Step 5: Generate information output
        return self.frequent_itemsets, self.rules

    def parse_file(self):
        """
        Parse the input file to extract transactions.

        Parameters:
        None

        Output:
        None (Populates self.transactions)
        """
        with open(self.file_path, "r") as file:
            lines = file.readlines()
        prev_transaction = -1
        for line in lines:
            item = line.split(" ")
            if prev_transaction != int(item[0]):
                prev_transaction = int(item[0])
                self.transactions.append(set())
            self.transactions[-1].add(item[1].replace("\n", ""))  # Add item to current transaction

if __name__ == "__main__":
    # Sample transactions for testing
    transactions = [
        {"milk", "bread", "nuts", "apple"},
        {"milk", "bread", "nuts"},
        {"milk", "bread"},
        {"milk", "bread", "apple"},
        {"bread", "apple"},
    ]
    
    min_support = 140  # Example threshold for support
    min_confidence = 0.8  # Example threshold for confidence
    input_file = "small.txt"  # Input file containing transaction data

    # Initialize Apriori algorithm and run it
    apriori = Apriori(min_support, min_confidence, input_file)
    frequent_itemsets, rules = apriori.run()
    
    # Print results
    print("Frequent Itemsets:", frequent_itemsets)
    print("Association Rules:", rules)
