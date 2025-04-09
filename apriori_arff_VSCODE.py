import arff
from itertools import combinations
from collections import defaultdict
import time
import matplotlib.pyplot as plt

def load_transactions(dataset):
    transactions = []
    
    with open(dataset, 'r') as file:
        data = arff.load(file)
        attributes = [attr[0] for attr in data['attributes']]  
        
        for row in data['data']:
            transaction = set(f"{attr}={value}" for attr, value in zip(attributes, row)) 
            transactions.append(transaction)
    
    return transactions

def generate_candidates(prev_frequent, k):
    candidates = set()
    prev_frequent = list(prev_frequent)  
    for i in range(len(prev_frequent)):
        for j in range(i + 1, len(prev_frequent)):
            merged = prev_frequent[i] | prev_frequent[j]  
            if len(merged) == k:
                candidates.add(frozenset(merged))
    return candidates

def prune_candidates(candidates, prev_frequent):
    pruned = set()
    for candidate in candidates:
        if all(frozenset(subset) in prev_frequent for subset in combinations(candidate, len(candidate) - 1)):
            pruned.add(candidate)
    return pruned

def count_support(transactions, candidates, min_support):
    counts = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                counts[candidate] += 1
    num_transactions = len(transactions)
    frequent = {itemset for itemset, count in counts.items() if count / num_transactions >= min_support}
    return frequent, counts

def apriori(dataset, min_support, min_confidence):
    transactions = load_transactions(dataset)
    k = 1
    frequent_itemsets = []
    single_items = {frozenset([item]) for transaction in transactions for item in transaction}
    f1, _ = count_support(transactions, single_items, min_support)
    fk = f1
    while fk:
        frequent_itemsets.extend(fk)
        candidates = generate_candidates(fk, k + 1)
        candidates = prune_candidates(candidates, fk)
        fk, _ = count_support(transactions, candidates, min_support)
        k += 1
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    num_transactions = len(transactions)
    rules = []
    support_data = {}
    for itemset in frequent_itemsets:
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        support_data[itemset] = count / num_transactions
    for itemset in frequent_itemsets:
        for i in range(1, len(itemset)):
            for subset in combinations(itemset, i):
                subset = frozenset(subset)
                remaining = itemset - subset
                if remaining:
                    confidence = support_data[itemset] / support_data[subset]
                    if confidence >= min_confidence:
                        rules.append((subset, remaining, confidence))
    return rules

dataset = "vote.arff"  
min_support_values = [i / 10 for i in range(1, 10)]
min_support = float(input("Enter minimum support (0-1): "))
min_confidence = float(input("Enter minimum confidence (0-1): "))

frequent_itemsets = apriori(dataset, min_support, min_confidence)
rules = generate_association_rules(frequent_itemsets, load_transactions(dataset), min_confidence)

print("Frequent Itemsets:")
for itemset in frequent_itemsets:
    print(", ".join(itemset))

print("\nAssociation Rules:")
for rule in rules:
    lhs = ", ".join(rule[0])  
    rhs = ", ".join(rule[1])  
    print(f"{lhs} -> {rhs}, confidence: {rule[2]:.2f}")

