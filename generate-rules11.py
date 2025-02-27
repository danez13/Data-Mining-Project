from itertools import combinations
class Apriori:
    def __init__(self, min_support, min_confidence):
        """
        Initialize the Apriori algorithm with minimum support and confidence thresholds.
        
        Parameters:
        min_support (float): The minimum support threshold.
        min_confidence (float): The minimum confidence threshold.
        
        Output:
        None
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.num_transactions = 0
        self.frequent_itemsets = {}  # Dictionary to store frequent itemsets with their support counts, organized by size
        self.rules = []  # List to store generated association rules
    
    def generate_candidates(self, prev_frequent_itemsets, k):
        """
        Generate candidate itemsets of size k from previous frequent itemsets.
        
        Parameters:
        prev_frequent_itemsets (set of frozensets): Frequent itemsets from the previous iteration.
        k (int): The size of itemsets to generate.
        
        Output:
        set of frozensets: Generated candidate itemsets.
        """
        candidates = set()
        for item in self.frequent_itemsets[1]:
            for itemset in prev_frequent_itemsets:
                if item.issubset(itemset):
                    continue
                candidates.add(itemset.union(item))
        return candidates
    
    def prune_candidates(self, candidates, prev_frequent_itemsets):
        """
        Prune candidates that contain non-frequent subsets.
        
        Parameters:
        candidates (set of frozensets): Candidate itemsets to be pruned.
        prev_frequent_itemsets (set of frozensets): Previous frequent itemsets for pruning reference.
        
        Output:
        set of frozensets: Pruned candidate itemsets.
        """
        pruned = set()
        prev_frequent_itemsets = set().union(*prev_frequent_itemsets)
        for candidate in candidates:
            if not candidate.issubset(prev_frequent_itemsets):
                continue
            pruned.add(candidate)
        return pruned
    
    def count_support(self, transactions, candidates):
        """
        Count the support of each candidate itemset in the transactions.
        
        Parameters:
        transactions (list of sets): Each set represents a transaction containing items.
        candidates (set of frozensets): Candidate itemsets whose support needs to be counted.
        
        Output:
        dict: Dictionary mapping candidate itemsets to their support counts.
        """
        support_counts = {}
        for candidate in candidates:
            support_count = 0
            for transaction in transactions:
                if candidate.issubset(transaction):
                    support_count+=1
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
        for candidate,support_count in candidates.items():
            if (support_count/self.num_transactions) < self.min_support:
                continue
            frequent_itemsets[candidate]=support_count
        return frequent_itemsets
    
    def find_frequent_itemsets(self, transactions):
        """
        Iteratively find frequent itemsets using the Apriori algorithm steps.
        
        Parameters:
        transactions (list of sets): Each set represents a transaction containing items.
        
        Output:
        None (Updates self.frequent_itemsets)
        
        Example:
        transactions = [
            {"milk", "bread", "nuts", "apple"},
            {"milk", "bread", "nuts"},
            {"milk", "bread"},
            {"milk", "bread", "apple"},
            {"bread", "apple"}
        ]
        """
        k = 1
        current_candidates_itemsets = {frozenset([item]) for transaction in transactions for item in transaction}
        support_counts = self.count_support(transactions, current_candidates_itemsets)
        
        self.frequent_itemsets = {}  # Set frequent itemsets dictionary
        self.frequent_itemsets[1] = self.eliminate_infrequent(support_counts)
        current_frequent_itemsets = self.frequent_itemsets[1]
        frequent_itemsets_k = self.frequent_itemsets[1]
        while current_frequent_itemsets:

            # generate candidate itemsets
            candidates = self.generate_candidates(set(frequent_itemsets_k.keys()), k)

            # pruning candidate itemsets
            current_candidates_itemsets = self.prune_candidates(candidates, set(frequent_itemsets_k.keys()))

            # Count support for current candidates
            support_counts = self.count_support(transactions, current_candidates_itemsets)

            # Eliminate infrequent itemsets
            frequent_itemsets_k = self.eliminate_infrequent(support_counts)

            # If empty, break
            # if not frequent_itemsets_k:
            #     break

            # Go to the next level
            k += 1
            if k == 3:
                break

            # Store frequent itemsets under their respective size category
            self.frequent_itemsets[k] = frequent_itemsets_k

            current_frequent_itemsets = frequent_itemsets_k
    def generate_item_output(self):
        with open("items11.txt","w") as file:
            for itemsets in self.frequent_itemsets.values():
                for itemset,support in itemsets.items():
                    file.write(f"{" ".join(map(str, itemset))}|{support}|{support/self.num_transactions}\n")

    def generate_rule_output(self):
        with open("rules11.txt","w") as file:
            for rule in self.rules:
                itemset = rule[0].split(",")+rule[1].split(",")
                set_size = len(frozenset(itemset))
                support = self.frequent_itemsets[set_size][frozenset(itemset)]
                file.write(f"{" ".join(map(str, rule[0].split(',')))}|{" ".join(map(str, rule[1].split(',')))}|{support}|{support/self.num_transactions}|{rule[2]}\n")
    def generate_info_output(self):
        pass
    
    def generate_rules(self):
        """
        Generate association rules from the discovered frequent itemsets.
        
        Parameters:
        None (Uses self.frequent_itemsets)
        
        Output:
        None (Updates self.rules)
        """
        # Iterate through each set of itemsets (1-itemsets, 2-itemsets, ...)
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
                        # Calculate confidence = support(antecedent ∪ consequent) / support(antecedent)
                        confidence = support / antecedent_support
                        
                        # If confidence meets the minimum threshold, add the rule
                        if confidence >= self.min_confidence:
                            rule = (*antecedent, *consequent, confidence)
                            self.rules.append(rule)

    def run(self, transactions):
        """
        Execute the Apriori algorithm: find frequent itemsets and generate rules.
        
        Parameters:
        transactions (list of sets): List of transactions containing itemsets.
        
        Output:
        tuple: (frequent_itemsets, rules)
        """
        self.num_transactions = len(transactions)
        self.find_frequent_itemsets(transactions)
        # self.generate_item_output()
        if self.min_confidence != -1:
            self.generate_rules()
            self.generate_rule_output()
        # self.generate_info_output()
        return self.frequent_itemsets, self.rules

def parse_file(file_path):
    with open(file_path,"r") as file:
        lines = file.readlines()
    transactions = []
    prev_tranaction = -1
    for line in lines:
        item = line.split(" ")
        if prev_tranaction != int(item[0]):
            prev_tranaction = int(item[0])
            transactions.append(set())
        transactions[-1].add(int(item[1]))
    return transactions

# Example usage and play data
if __name__ == "__main__":
    transactions = [
        {"milk", "bread", "nuts", "apple"},
        {"milk", "bread", "nuts"},
        {"milk", "bread"},
        {"milk", "bread", "apple"},
        {"bread", "apple"},
    ]
    
    min_support = 0.5  # Example threshold
    min_confidence = 0.7  # Example threshold
    input_file = "small.txt"
    # transactions = parse_file(input_file)

    apriori = Apriori(min_support, min_confidence)
    frequent_itemsets, rules = apriori.run(transactions)
    
    # print("Frequent Itemsets:", frequent_itemsets)
    # print("Association Rules:", rules)

    ''' Toy Example Output`  :
	Frequent Itemsets: {1: {
                            frozenset({'milk'}): 4,
                            frozenset({'bread'}): 5,
                            frozenset({'apple'}): 3
                            }, 
                        2: {
                            frozenset({'milk', 'bread'}): 3,
                            frozenset({'bread', 'apple'}): 3}
                            }

	Association Rules: [('milk','bread', 0.75), ('apple','bread', 1.0)]
    '''
