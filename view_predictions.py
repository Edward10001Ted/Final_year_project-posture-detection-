#!/usr/bin/env python3
"""View saved predictions from predictions_log.csv"""
import csv
import os

log_file = 'predictions_log.csv'

if not os.path.exists(log_file):
    print(f"No predictions log found at {log_file}")
    exit(1)

print(f"\n{'Timestamp':<30} {'Label':<10} {'Score':<8}")
print("-" * 50)

with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        timestamp = row['timestamp'][:19]  # trim to YY-MM-DD HH:MM:SS
        label = row['label']
        score = f"{float(row['score']):.4f}"
        print(f"{timestamp:<30} {label:<10} {score:<8}")
        count += 1

print(f"\nTotal predictions logged: {count}")
