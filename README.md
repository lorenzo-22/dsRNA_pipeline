# dsRNA Worst-Case Off-Target Pipeline (v2)

[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Pipeline-blue.svg)](https://github.com/lorenzo-22/dsRNA_pipeline)
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
git clone https://github.com/lorenzo-22/dsRNA_pipeline.git
cd dsRNA_pipeline
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

### 1. Configure Gene IDs
Update `input/gene_ids.txt` with your target genes. The format supports specifying multiple window sizes:
Format: `<Any_ID>,<Gene_Description>,<Window_Size_1>[-<Window_Size_2>...]`

Example:
```
TC015612,Acetyl-CoA carboxylase,300
TC000144,Proteasome subunit alpha type-6,300-490
```
- For the first gene, it will run analysis with a 300bp window.
- For the second gene, it will run analysis for both 300bp and 490bp windows.
- If no window size is specified, it defaults to 300bp.

---

## 📖 Usage

After activating the environment, use the `dsrna-pipeline` command to run the steps.

### 🚀 Running the Full Pipeline
The `run-all` command executes steps 1 through 6 in sequence for all specified window sizes. **Note:** It does *not* include the final `aggregate` step.

```bash
dsrna-pipeline run-all -i input/gene_ids.txt --reference "Phaedon cochleariae"
```

---

## 📂 Reorganized Output Structure

The pipeline now organizes results by window size to allow multi-scale analysis:

```
output/
└── Organisms/
    └── Phaedon_cochleariae/
        └── <Gene_Name>/
            ├── alignments/
            │   ├── msa/          # Base MSA files (window independent)
            │   └── pairwise/     # Base pairwise files (window independent)
            └── <Window_Size>/    # (e.g., 300, 490)
                ├── similarity/
                │   ├── msa/      # % Identity plots for this window
                │   └── pairwise/ # % Identity plots for this window
                ├── accessibility/
                ├── bowtie_matches/
                └── summary/
                    ├── pipeline_summary_metrics.png
                    └── top_10_worst_case_windows.csv
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
**Lab:** Center for non-coding RNA in Technology and Health (RTH), University of Copenhagen  
**License:** MIT
