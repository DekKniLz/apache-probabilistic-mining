# Project Overview

This project contains the automated data pipeline for researching **Probabilistic Data Structures** within the Apache Software Foundation. It extracts, parses, and analyzes Abstract Syntax Trees (ASTs) from official Apache Java repositories to understand the adoption, architectural domains, implementation origins, and technical debt of structures such as **Bloom Filters** and **HyperLogLog**.

# System Architecture

The pipeline is divided into three automated phases:

1. **Extraction Phase (`fetch_apache_repos.py`)**
   - Retrieves the official project registry from Apache.
   - Cross-references projects with the GitHub API.
   - Isolates active Java repositories.

2. **Mining Phase (`miner.py`)**
   - Uses the `tree-sitter` library to scan downloaded source code.
   - Locates precise implementations and imports of probabilistic data structures.

3. **Analytics Phase (`analytics.py`)**
   - Cleans the extracted data.
   - Infers usage domains.
   - Generates final statistics and visualizations.

# Prerequisites and Setup

To run this project, you need **Python 3.8 or higher**.

## Required Libraries

- `requests`
- `python-dotenv`
- `pandas`
- `matplotlib`
- `seaborn`
- `tree-sitter`
- `tree-sitter-java`

Example installation:

```bash
pip install requests python-dotenv pandas matplotlib seaborn tree-sitter tree-sitter-java
```

## Environment Configuration

Create a `.env` file in the project root directory and add your GitHub token:

```env
GITHUB_TOKEN=your_token_here
```

This token helps prevent GitHub API rate limits during the extraction phase.

# How to Use the Pipeline

Ensure your virtual environment is active, then run:

```bash
python3 main.py
```

The orchestrator will automatically:

- Run the extraction process.
- Clone the repositories.
- Parse the source code.
- Generate datasets and visualizations.

> **Note:** Downloading and analyzing large Apache repositories (such as Hadoop or Cassandra) may take several hours depending on your internet connection and hardware specifications.

# Output Files

## Repository List

- `repos.txt`
  - Contains the validated list of Apache Java repositories.

## Datasets

Located in the `output/` directory:

- `dataset.csv`
  - Raw data extracted from source code.

- `dataset_enriched.json`
  - Cleaned and enriched dataset ready for future analysis.

## Visualizations

Located in the `plots/` directory:

- Four generated charts that answer the core research questions of the study.
