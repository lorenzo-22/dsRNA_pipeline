# dsRNA Worst-Case Off-Target Pipeline (v2)

[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Pipeline-blue.svg)](https://github.com/rth-ensafe/dsrna-worst-case-pipeline)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized bioinformatics pipeline designed to identify **worst-case off-target scenarios** for dsRNA-based pesticides. By analyzing ortholog conservation, k-mer matching (Bowtie2), and RNA accessibility (ViennaRNA), this tool highlights genomic regions with the highest risk of cross-species activity in non-target organisms (NTOs).

---

## 🚀 Overview

The pipeline identifies regions in a target gene that are most likely to elicit off-target effects in Non-Target Organisms (NTOs) by searching for:
1.  **High Sequence Similarity**: Regions with the highest pairwise identity across arthropod orthologs.
2.  **High Accessibility**: Regions where the NTO mRNA is most likely to be unpaired and accessible for siRNA binding.
3.  **High k-mer Matches**: Regions with the maximum number of exact (0mm) and near (1mm) 21-mer matches in NTO genomes.

---

## 🛠 Installation

### Prerequisites
- **Python 3.12+**
- **Conda** or **Mamba** (recommended)
- **HPC Environment** with Slurm (optional, for large-scale analysis)

### Step 1: Clone the Repository
```bash
git clone https://github.com/rth-ensafe/dsrna-worst-case-pipeline.git
cd dsrna-worst-case-pipeline
```

### Step 2: Install Dependencies
We recommend using `uv` for lightning-fast installation, but `pip` also works:

```bash
# Using uv (recommended)
uv sync

# Activate the virtual environment
source .venv/bin/activate       # bash/zsh
source .venv/bin/activate.fish  # fish
source .venv/bin/activate.csh   # csh

# Now the 'dsrna-pipeline' command is available directly
```

### Step 3: External Tools
Ensure the following tools are available in your `$PATH` or loaded via modules:
- **Bowtie2** (v2.4.5+)
- **EMBOSS** (for `needle` alignment)
- **Clustal Omega** (for MSA)
- **ViennaRNA** (Python bindings `RNA`)

---

## 📖 Usage

After activating the environment, use the `dsrna-pipeline` command to run the steps.

### 1. Download Orthologs
Provide a list of gene descriptions and IDs to fetch CDS sequences from OrthoDB.
```bash
dsrna-pipeline fetch-cds -i input/gene_ids.txt -o output/orthologs -t 6656
```

### 2. Alignment (MSA & Pairwise)
Generate multiple sequence alignments and pairwise comparisons to the reference species.
```bash
dsrna-pipeline align run-msa --reference "Phaedon cochleariae"
```

### 3. K-mer Matching (Bowtie2)
Analyze 21-mer conservation across NTOs to find potential off-target seeds.
```bash
dsrna-pipeline bowtie run-all --reference "Phaedon cochleariae"
```

### 4. RNA Accessibility
Calculate the probability of nucleotides being unpaired using ViennaRNA.
```bash
dsrna-pipeline accessibility run-all --reference "Phaedon cochleariae"
```

### 5. Aggregate & Identify Worst-Case
The final step aggregates all metrics and identifies the **Top 10 Worst-Case Windows** (highest risk).
```bash
dsrna-pipeline aggregate run-all --reference "Phaedon cochleariae"
```

---

## 📊 Worst-Case Selection Logic

The pipeline prioritizes risk by sorting 300bp windows based on:
1.  **Avg_Pairwise_Identity** (Descending): Maximizes similarity to NTOs.
2.  **Avg_NTO_Accessibility** (Descending): Maximizes the chance of siRNA binding in NTOs.
3.  **Bowtie_Hits_0mm** (Descending): Maximizes exact 21-mer matches.
4.  **Bowtie_Hits_1mm** (Descending): Maximizes near-matches.

The results are saved as `top_10_worst_case_windows.csv` for each gene.

---

## 📂 Output Structure

```
output/
└── Organisms/
    └── Phaedon_cochleariae/
        └── <Gene_Name>/
            ├── summary/
            │   ├── pipeline_summary_metrics.png   # 5-panel risk visualization
            │   └── top_10_worst_case_windows.csv  # The high-risk targets
            ├── pairwise_alignments/
            ├── accessibility/
            └── bowtie_matches/
```

---

## ✉️ Contact & Support

**Author:** Lorenzo Favaro  
**Lab:** RNA Therapeutics and Hygiene (RTH), University of Copenhagen  
**License:** MIT
