# DGA Classifier

Classifies domains as algorithmically generated (DGA) or benign using
character-level features (entropy, length, character runs). Goal: score
Pi-hole DNS query logs.

## Setup
    pip install pandas scikit-learn tldextract
    git clone https://github.com/baderj/domain_generation_algorithms.git

Download the Tranco top-1M list from https://tranco-list.eu and unzip it here.

## Run (from this folder)
    python dga_functions/generate_dga.py --repo domain_generation_algorithms   # -> dga_domains.csv
    python dga_functions/builder.py --dga dga_domains.csv --benign top-1m.csv   # -> dataset.csv
    python dga_functions/trainer.py --data dataset.csv

The trainer uses leave-one-family-out evaluation: each DGA family is tested
after training only on the others, so the scores show how well the model
catches families it has never seen.

DGA implementations: https://github.com/baderj/domain_generation_algorithms
