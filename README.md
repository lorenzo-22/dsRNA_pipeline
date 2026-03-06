# dsRNA Conservation Pipeline - Complete Guide

**Author:** Lorenzo Favaro  
**Date:** December 2025  
**Purpose:** Analyze ortholog conservation across arthropod species to identify worst-case dsRNA off-target scenarios

---

## Quick Start

```bash
# Navigate to pipeline directory
cd /home/projects/rth/ENSAFE/projects/siRNA/projects/subproject1/dsRNA_worst_case_pipeline_v2

# Follow steps 1-6 below
```

---

## Overview

This pipeline analyzes conservation of genes across multiple arthropod species to identify regions with high cross-species similarity (worst-case for dsRNA specificity). The workflow:

1. **Download orthologs** from OrthoDB
2. **Align sequences** with MSA
3. **Analyze k-mer conservation** with Bowtie2
4. **Compute RNA accessibility**
5. **Generate integrated visualizations**

**Expected time:** 2-4 hours for ~30 genes (depending on cluster queue)

---

## Step 1: Download Ortholog Sequences

### Input Files

You need an input file in the `input/` directory:

**1. Gene IDs file:** `input/gene_ids.txt`

Format: `<Any_ID>,<Gene_Description>`

```
TC015612,Acetyl-CoA carboxylase
TC000069,Proteasome subunit beta type-1
TC000144,Proteasome subunit alpha type-6
```

**Important:** The first column can be any identifier (doesn't need to be OrthoDB cluster IDs). The script will search OrthoDB using the gene description (second column).

### Download Sequences

Run the OrthoDB fetcher script to download CDS sequences for all genes:

```bash
python3 src/orthodb_fetch_cds.py \
    -i input/gene_ids.txt \
    -o output/orthologs \
    -t 6656
```

**Parameters:**
- `-i, --ids-file`: Your gene IDs file (format: `ID,Description` per line)
- `-o, --outdir`: Where to save FASTA files
- `-t, --taxon`: NCBI taxonomy ID (6656 = Arthropoda)

**What happens:**
1. Script reads each line from `gene_ids.txt` (format: `ID,Gene Description`)
2. **Searches OrthoDB by gene description first** (primary strategy)
3. Falls back to using ID if description search fails
4. Retrieves CDS sequences from all arthropod species
5. Saves as `<ID>_cds_arthropoda.fa` in output directory

**Note:** The script searches by **gene description** (e.g., "Acetyl-CoA carboxylase"), so your IDs don't need to be OrthoDB cluster IDs. Any species-specific IDs (TC*, TG*, etc.) will work as long as you provide the gene description.

### Check Output

```bash
# List downloaded FASTA files
ls -lh output/orthologs/

# Count sequences in a file
grep -c "^>" output/orthologs/TC000069_cds_arthropoda.fa

# View first few sequences
head -n 20 output/orthologs/TC000069_cds_arthropoda.fa
```

Expected output structure:
```
output/
├── orthologs/
│   ├── TC000069_cds_arthropoda.fa
│   ├── TC000144_cds_arthropoda.fa
│   ├── TC006375_cds_arthropoda.fa
│   └── ... (one FASTA per gene)
└── missing_ids.txt (if any genes failed)
```

### Troubleshooting Step 1

**Problem:** `No sequences returned for <gene_id>`

**Solutions:**
```bash
# Check if gene ID is valid in OrthoDB
# Visit: https://www.orthodb.org/?query=<gene_id>

# Try with description search (automatic fallback)
# The script will search by gene description if ID fails

# Force REST API instead of Python client
python3 src/orthodb_fetch_cds.py \
    --ids-file input/gene_ids.txt \
    --outdir output/orthologs \
    --force-rest
```

**Problem:** Script crashes or hangs

**Solutions:**
```bash
# Check internet connection
ping www.orthodb.org

# Run with verbose output
python3 -u src/orthodb_fetch_cds.py --ids-file input/gene_ids.txt --outdir output/orthologs

# Process one gene at a time
head -1 input/gene_ids.txt > test_gene.txt
python3 src/orthodb_fetch_cds.py --ids-file test_gene.txt --outdir output/orthologs
```

**Problem:** Some genes are missing

```bash
# Check the missing IDs report
cat output/missing_ids.txt

# Manually search for these genes in OrthoDB
# Update gene_ids.txt with correct IDs
```

---

## What's Next?

After Step 1 completes successfully:

✅ You have FASTA files for all genes in `output/orthologs/`  
✅ Each FASTA contains CDS sequences from multiple arthropod species  
✅ Ready to proceed to Step 2: Multiple Sequence Alignment

**Continue to:** [Step 2: Multiple Sequence Alignment](#step-2-multiple-sequence-alignment) (coming next)

---

## Step 2: Multiple Sequence Alignment

### Filter Sequences by Species

Create gene-specific directories and filter sequences to include only your target species:

```bash
python3 src/create_dirs_filter_run_msa.py \
    --ids-file input/gene_ids.txt \
    --in-fasta-dir output/orthologs \
    --insects-csv input/insects_list.csv \
    --outdir output/genes \
    --create-plots
```

**What happens:**
1. Creates a directory for each gene (named after description)
2. Filters FASTA to keep only species from `insects_list.csv`
3. Writes filtered FASTA as `TC<ID>_filtered.fa`
4. Prepares for MSA alignment

**Check output:**
```bash
# List gene directories
ls -d output/genes/*/

# Check filtered sequences
head output/genes/Proteasome_subunit_beta_type-1/TC000069_filtered.fa

# Count species in filtered file
grep -c "^>" output/genes/Proteasome_subunit_beta_type-1/TC000069_filtered.fa
```

### Standardize Sequence IDs

Clean up FASTA headers for consistent processing:

```bash
python3 src/standardize_sequence_ids.py --all
```

This ensures all sequence IDs follow the same format for downstream analysis.

### Run Multiple Sequence Alignments

#### Option A: Interactive (one gene at a time)

```bash
# Run MSA for a single gene
bash src/run_msa.sh \
    output/genes/Proteasome_subunit_beta_type-1/TC000069_filtered.fa \
    "Apis mellifera"
```

Parameters:
- First argument: Path to filtered FASTA
- Second argument: Reference organism name (must exist in FASTA)

#### Option B: SLURM Batch (all genes)

For cluster execution:

```bash
# Load required module
module load clustal-omega/1.2.4

# Submit array job for all genes
sbatch --array=0-34 --mem=8G --ntasks=4 src/run_msa.sh

# Monitor jobs
squeue -u $USER

# Check specific job
scontrol show job <JOBID>
```

**Check MSA output:**
```bash
# View alignment file
head output/genes/Proteasome_subunit_beta_type-1/TC000069_filtered_clustalw.aln

# Check log
cat slurm_logs/msa_*.out
```

### Compute Sliding Window Conservation

After MSA completes, analyze conservation in 300 bp windows:

#### MSA-based conservation:
```bash
python3 src/compute_window_msa.py \
    --msa-root output/genes \
    --all \
    --analysis-window 300
```

#### Pairwise conservation:
```bash
# Submit SLURM job for all genes
sbatch --array=0-34 --mem=8G --ntasks=4 src/run_pairwise.sh

# OR run locally
python3 src/compute_window_pairwise.py \
    --msa-root output/genes \
    --all \
    --analysis-window 300
```

**Check conservation output:**
```bash
# View window scores
head output/genes/Proteasome_subunit_beta_type-1/windows_conservation_msa.csv
head output/genes/Proteasome_subunit_beta_type-1/windows_conservation_pairwise.csv
```

---

## Step 3: K-mer Conservation Analysis

### Prepare Reference and Build Index

For each gene, select a reference organism and build Bowtie2 index:

```bash
# Load Bowtie2
module load bowtie2

# Run for all genes
sbatch --array=0-34 --mem=8G --ntasks=4 src/run_prepare_reference_all_genes.sh
```

**What happens:**
1. Extracts reference sequence (e.g., "Phaedon cochleariae")
2. Generates 21-mers from reference
3. Builds Bowtie2 index of other species' sequences

**Check output:**
```bash
# View preparation output
ls output/genes/Proteasome_subunit_beta_type-1/prep/

# Check reference sequence
head output/genes/Proteasome_subunit_beta_type-1/prep/reference.fa

# Check k-mers
head output/genes/Proteasome_subunit_beta_type-1/prep/reference_21mers.fa
```

### Align K-mers with Bowtie2

Align reference k-mers to all other organisms:

```bash
sbatch --array=0-34 src/run_kmer_alignment_all_genes.sh
```

**Check alignment output:**
```bash
# View SAM file
head output/genes/Proteasome_subunit_beta_type-1/prep/alignment_results.sam

# Count alignments
grep -v "^@" output/genes/Proteasome_subunit_beta_type-1/prep/alignment_results.sam | wc -l
```

### Analyze Matches by Organism

Extract k-mer matches per organism:

```bash
bash src/run_organism_analysis_all_genes.sh
```

### Aggregate K-mer Windows

Combine k-mer data into 300 bp windows:

```bash
sbatch --array=0-34 src/run_aggregate_kmer_all_genes.sh
```

**Check window output:**
```bash
head output/genes/Proteasome_subunit_beta_type-1/kmer_windows_by_organism.csv
```

---

## Step 4: RNA Accessibility

Compute and plot RNA accessibility (requires pre-computed RNAplfold data):

```bash
# Load ViennaRNA
module load RNAfold

# Plot accessibility for all genes
sbatch src/run_plot_accessibility_all_genes.sh
```

**Check output:**
```bash
ls output/genes/*/accessibility_plot.png
```

---

## Step 5: Generate Final Visualizations

Combine all metrics (MSA conservation, k-mer conservation, accessibility):

```bash
# Run for all genes
bash src/run_aggregate_all_genes.sh
```

**Final outputs:**
```bash
# View integrated plots
ls output/genes/*/combined_metrics_plot.png

# Open in browser or image viewer
display output/genes/Proteasome_subunit_beta_type-1/combined_metrics_plot.png
```

---

## Complete Pipeline Summary

Quick reference for running the entire pipeline:

```bash
# Step 1: Download orthologs
python3 src/orthodb_fetch_cds.py -i input/gene_ids.txt -o output/orthologs -t 6656

# Step 2: MSA
python3 src/create_dirs_filter_run_msa.py --ids-file input/gene_ids.txt --in-fasta-dir output/orthologs --insects-csv input/insects_list.csv --outdir output/genes
python3 src/standardize_sequence_ids.py --all
sbatch --array=0-34 src/run_msa.sh
python3 src/compute_window_msa.py --msa-root output/genes --all --analysis-window 300
sbatch --array=0-34 src/run_pairwise.sh

# Step 3: K-mer analysis
module load bowtie2
sbatch --array=0-34 src/run_prepare_reference_all_genes.sh
sbatch --array=0-34 src/run_kmer_alignment_all_genes.sh
bash src/run_organism_analysis_all_genes.sh
sbatch --array=0-34 src/run_aggregate_kmer_all_genes.sh

# Step 4: Accessibility
sbatch src/run_plot_accessibility_all_genes.sh

# Step 5: Final plots
bash src/run_aggregate_all_genes.sh
```

---

## Directory Structure After Completion

```
dsRNA_worst_case_pipeline_v2/
├── input/
│   ├── gene_ids.txt
│   └── insects_list.csv
├── output/
│   ├── orthologs/                  # Step 1 output
│   │   └── TC*_cds_arthropoda.fa
│   └── genes/                      # Steps 2-5 output
│       └── <Gene_Description>/
│           ├── TC*_filtered.fa
│           ├── TC*_filtered_clustalw.aln
│           ├── windows_conservation_msa.csv
│           ├── windows_conservation_pairwise.csv
│           ├── prep/
│           │   ├── reference.fa
│           │   ├── reference_21mers.fa
│           │   └── alignment_results.sam
│           ├── organism_matches/
│           ├── kmer_windows_by_organism.csv
│           ├── accessibility_plot.png
│           └── combined_metrics_plot.png
└── src/                            # All scripts
```

---

## Monitoring Jobs

```bash
# Check all your jobs
squeue -u $USER

# Watch jobs in real-time
watch -n 5 'squeue -u $USER'

# Check specific job details
scontrol show job <JOBID>

# View logs
tail -f slurm_logs/*.out
tail -f slurm_logs/*.err

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

---

## Common Issues

### Module not found

```bash
# List available modules
module avail

# Load specific version
module load clustal-omega/1.2.4
module load bowtie2/2.4.5
```

### Array index out of range

```bash
# Count genes
NUM_GENES=$(wc -l < input/gene_ids.txt)

# Adjust array (0-indexed)
sbatch --array=0-$((NUM_GENES-1)) src/run_msa.sh
```

### Out of memory

```bash
# Increase memory
sbatch --array=0-34 --mem=16G src/run_msa.sh

# Or reduce parallelism
sbatch --array=0-34%10 --mem=8G src/run_msa.sh  # max 10 concurrent
```

---

## Tips

- **Test first**: Run on 1-2 genes before full pipeline
- **Check logs**: Always examine SLURM output/error files
- **Monitor resources**: Use `seff <JOBID>` to check resource usage
- **Incremental**: Each step can run independently if previous completed

---

## Contact

**Lorenzo Favaro**  
Email: lorenzo@rth.dk  
RNA Therapeutics and Hygiene (RTH)  
University of Copenhagen
