import json
import sys
import subprocess
import os
import math
from pathlib import Path
from typing import Optional, List

import httpx
import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import seaborn as sns
from pymsaviz import MsaViz

app = typer.Typer(help="Fetch CDS sequences from OrthoDB and analyze them (length, alignment, information content).")

# Configure loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{message}</cyan>", level="INFO")

ORTHODB_BASE_URL = "https://data.orthodb.org/current"
DEFAULT_TAXON = "6656"  # Arthropoda


def fetch_orthodb_data(endpoint: str, params: dict) -> Optional[dict]:
    """Helper to fetch data from OrthoDB REST API."""
    url = f"{ORTHODB_BASE_URL}/{endpoint}"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching data from {url}: {e}")
        return None


def fetch_fasta(cluster_id: str, taxon: str, seq_type: str = "cds") -> Optional[str]:
    """Fetch FASTA sequences for a given cluster and taxon."""
    url = f"{ORTHODB_BASE_URL}/fasta"
    params = {
        "id": cluster_id,
        "species": taxon,
        "seqtype": seq_type
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error(f"Error fetching FASTA for cluster {cluster_id}: {e}")
        return None


def find_cluster_for_gene(gene_id: str, gene_name: str, taxon: str) -> Optional[str]:
    """
    Find the OrthoDB cluster ID for a given gene ID or name at a specific taxonomic level.
    """
    logger.debug(f"Searching for gene ID: {gene_id}")
    search_data = fetch_orthodb_data("search", {"query": gene_id, "level": taxon})
    
    if not search_data or not search_data.get("data"):
        logger.debug(f"Gene ID {gene_id} not found, searching for gene name: {gene_name}")
        search_data = fetch_orthodb_data("search", {"query": gene_name, "level": taxon})

    if not search_data or not search_data.get("data"):
        logger.debug(f"Searching for gene name: {gene_name} without level constraint")
        search_data = fetch_orthodb_data("search", {"query": gene_name})

    if not search_data or not search_data.get("data"):
        return None

    if search_data.get("bigdata"):
        for entry in search_data["bigdata"]:
            cluster_id = entry.get("id")
            if cluster_id and f"at{taxon}" in cluster_id:
                return cluster_id
        return search_data["bigdata"][0].get("id")

    return search_data["data"][0]


def calculate_information_content(alignment_file: Path) -> pd.DataFrame:
    """
    Calculate Information Content (IC) using Shannon entropy and 
    Schneider & Stephens small-sample correction.
    """
    records = list(SeqIO.parse(alignment_file, "fasta"))
    if not records:
        return pd.DataFrame()

    aln_len = len(records[0].seq)
    n_seq = len(records)
    s = 4  # A, C, G, T
    h_max = math.log2(s)
    ln2 = math.log(2)

    ic_results = []

    for i in range(aln_len):
        col = [str(r.seq[i]).upper() for r in records]
        # Only count valid bases for entropy calculation
        valid_bases = [b for b in col if b in 'ACGTU']
        n = len(valid_bases)
        
        if n == 0:
            ic_results.append({"Position": i + 1, "IC": 0.0, "GapFraction": 1.0})
            continue

        counts = {b: valid_bases.count(b) for b in 'ACGT'}
        # Handle Uracil if present
        if 'U' in valid_bases:
            counts['T'] += valid_bases.count('U')

        # Shannon Entropy H(l)
        h_l = 0
        for b in 'ACGT':
            p = counts[b] / n
            if p > 0:
                h_l -= p * math.log2(p)

        # Schneider & Stephens small-sample correction factor e(n)
        e_n = (s - 1) / (2 * n * ln2)
        
        # Information Content R_seq(l) = H_max - (H(l) + e(n))
        r_l = h_max - (h_l + e_n)
        
        # IC shouldn't be negative (can happen if correction factor is larger than H_max - H(l))
        ic_results.append({
            "Position": i + 1, 
            "IC": max(0, r_l),
            "GapFraction": (n_seq - n) / n_seq
        })

    return pd.DataFrame(ic_results)


@app.command()
def fetch_cds(
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input", "-i", help="Input file with gene IDs and names (CSV/TXT)."),
    output_dir: Path = typer.Option(Path("output/orthologs"), "--output", "-o", help="Output directory for FASTA files."),
    species_file: Optional[Path] = typer.Option(Path("input/insects_list.csv"), "--species-list", "-s", help="Optional CSV file with species to keep."),
    taxon: str = typer.Option(DEFAULT_TAXON, "--taxon", "-t", help="NCBI Taxonomy ID for filtering (default: 6656 for Arthropoda)."),
):
    """
    Fetch CDS sequences for a list of genes.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise typer.Exit(code=1)

    target_species = None
    if species_file and species_file.exists():
        try:
            species_df = pd.read_csv(species_file)
            col = "Species" if "Species" in species_df.columns else species_df.columns[0]
            target_species = set(species_df[col].astype(str).str.strip().tolist())
            logger.info(f"Loaded {len(target_species)} target species for filtering.")
        except Exception as e:
            logger.error(f"Error reading species list: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    try:
        lines = input_file.read_text().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = [s.strip() for s in line.split(",", 1)]
                data.append(parts)
            else:
                val = line.strip()
                data.append([val, val])
        
        df = pd.DataFrame(data, columns=["gene_id", "gene_name"])
        logger.info(f"Loaded {len(df)} genes from {input_file.name}")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        raise typer.Exit(code=1)

    logger.info(f"Processing {len(df)} genes...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching sequences"):
        gene_id = row["gene_id"]
        gene_name = row["gene_name"]
        
        cluster_id = find_cluster_for_gene(gene_id, gene_name, taxon)
        
        if not cluster_id:
            logger.warning(f"Could not find cluster for {gene_id} ({gene_name})")
            continue
            
        fasta_content = fetch_fasta(cluster_id, taxon)
        
        if fasta_content and fasta_content.strip():
            if target_species:
                filtered_records = []
                parts = fasta_content.split(">")
                for part in parts:
                    if not part.strip():
                        continue
                    lines = part.split("\n", 1)
                    header = lines[0]
                    try:
                        start = header.find("{")
                        end = header.rfind("}")
                        if start != -1 and end != -1:
                            meta_str = header[start : end + 1]
                            meta = json.loads(meta_str)
                            org_name = meta.get("organism_name", "").strip()
                            if org_name in target_species:
                                filtered_records.append(f">{part.strip()}")
                    except Exception as e:
                        logger.debug(f"Could not parse header metadata: {header} - {e}")
                fasta_content = "\n".join(filtered_records) + "\n" if filtered_records else ""

            if fasta_content.strip():
                safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in gene_name.replace(" ", "_"))
                output_file = output_dir / f"{safe_name}_{gene_id}_{cluster_id}.fasta"
                output_file.write_text(fasta_content)
                logger.info(f"Saved {len(fasta_content)} bytes to {output_file}")
            else:
                logger.warning(f"No sequences matched target species for cluster {cluster_id}")
        else:
            logger.warning(f"No FASTA content found for cluster {cluster_id}")


@app.command()
def plot_lengths(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i", help="Directory containing filtered FASTA files."),
    plot_dir: Path = typer.Option(Path("output/plots"), "--output", "-o", help="Directory to save plots."),
):
    """
    Generate CDS length distribution plots for each organism and gene.
    """
    if not fasta_dir.exists():
        logger.error(f"FASTA directory not found: {fasta_dir}")
        raise typer.Exit(code=1)

    plot_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = list(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        logger.warning(f"No FASTA files found in {fasta_dir}")
        return

    all_data = []

    logger.info(f"Analyzing {len(fasta_files)} FASTA files for length distribution...")

    for fasta_file in tqdm(fasta_files, desc="Parsing FASTA files"):
        parts = fasta_file.stem.rsplit("_", 2)
        gene_name = parts[0] if len(parts) == 3 else fasta_file.stem
        
        try:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            for record in records:
                header = record.description
                start = header.find("{")
                end = header.rfind("}")
                if start != -1 and end != -1:
                    meta = json.loads(header[start : end + 1])
                    org_name = meta.get("organism_name", "Unknown")
                    all_data.append({
                        "Gene": gene_name,
                        "Organism": org_name,
                        "Length": len(record.seq),
                        "File": fasta_file.name
                    })
        except Exception as e:
            logger.error(f"Error parsing {fasta_file.name}: {e}")

    if not all_data:
        logger.error("No valid sequence data found for plotting.")
        return

    df = pd.DataFrame(all_data)
    unique_genes = df["Gene"].unique()
    
    logger.info(f"Generating plots for {len(unique_genes)} genes...")
    
    for gene in tqdm(unique_genes, desc="Generating gene plots"):
        gene_df = df[df["Gene"] == gene]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=gene_df, x="Organism", y="Length")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"CDS Length Distribution for {gene}")
        plt.tight_layout()
        out_path = plot_dir / f"{gene}_length_distribution.png"
        plt.savefig(out_path)
        plt.close()

    logger.info("Generating summary comparison plot...")
    plt.figure(figsize=(14, 8))
    pivot_df = df.groupby(["Gene", "Organism"], observed=True)["Length"].mean().reset_index()
    sns.barplot(data=pivot_df, x="Gene", y="Length", hue="Organism")
    plt.xticks(rotation=45, ha='right')
    plt.title("Average CDS Length per Gene and Organism")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    summary_path = plot_dir / "summary_length_comparison.png"
    plt.savefig(summary_path)
    plt.close()
    logger.success("Plotting completed successfully.")


@app.command()
def align_sequences(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i", help="Directory containing filtered FASTA files."),
    align_dir: Path = typer.Option(Path("output/alignments"), "--output", "-o", help="Directory to save alignments and plots."),
    slurm: bool = typer.Option(True, help="Use SLURM to submit alignment jobs."),
    cpus_per_task: int = typer.Option(4, help="CPUs per task for SLURM."),
    mem: str = typer.Option("8G", help="Memory for SLURM job."),
    time: str = typer.Option("02:00:00", help="Time limit for SLURM job."),
):
    """
    Perform Multiple Sequence Alignment (MSA) using Clustal Omega and plot results.
    """
    if not fasta_dir.exists():
        logger.error(f"FASTA directory not found: {fasta_dir}")
        raise typer.Exit(code=1)

    align_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir = align_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = list(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        logger.warning(f"No FASTA files found in {fasta_dir}")
        return

    logger.info(f"Preparing alignment for {len(fasta_files)} genes...")

    for fasta_file in tqdm(fasta_files, desc="Processing genes"):
        parts = fasta_file.stem.rsplit("_", 2)
        gene_name = parts[0] if len(parts) == 3 else fasta_file.stem
        
        try:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            if len(records) < 2:
                logger.warning(f"Skipping {gene_name}: less than 2 sequences.")
                continue

            temp_input = align_dir / f"{fasta_file.stem}_renamed.fasta"
            aln_file = align_dir / f"{fasta_file.stem}_aligned.fasta"
            plot_path = align_dir / f"{fasta_file.stem}_alignment.png"
            ic_plot_path = align_dir / f"{fasta_file.stem}_information_content.png"
            ic_csv_path = align_dir / f"{fasta_file.stem}_information_content.csv"

            renamed_records = []
            for i, r in enumerate(records):
                try:
                    header = r.description
                    start = header.find("{")
                    end = header.rfind("}")
                    if start != -1 and end != -1:
                        meta = json.loads(header[start : end + 1])
                        org_name = meta.get("organism_name", r.id).replace(" ", "_")
                    else:
                        org_name = r.id
                except:
                    org_name = r.id
                new_id = f"{org_name}_{i}"
                renamed_records.append(SeqRecord(r.seq, id=new_id, description=""))
            SeqIO.write(renamed_records, temp_input, "fasta")

            if slurm:
                job_script = slurm_dir / f"align_{fasta_file.stem}.sh"
                job_name = f"msa_{gene_name[:10]}"
                script_path = Path(__file__).resolve()
                
                content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_dir}/{fasta_file.stem}.out
#SBATCH --error={slurm_dir}/{fasta_file.stem}.err
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}

module load clustal-omega

clustalo -i {temp_input.resolve()} -o {aln_file.resolve()} --force --outfmt=fasta --threads={cpus_per_task}

{sys.executable} {script_path} internal-plot {aln_file.resolve()} {plot_path.resolve()} {ic_plot_path.resolve()} {ic_csv_path.resolve()} "{gene_name}"
"""
                job_script.write_text(content)
                subprocess.run(["sbatch", str(job_script)], check=True)
            else:
                subprocess.run(["bash", "-c", f"module load clustal-omega && clustalo -i {temp_input} -o {aln_file} --force --outfmt=fasta"], check=True)
                internal_plot(aln_file, plot_path, ic_plot_path, ic_csv_path, gene_name)

        except Exception as e:
            logger.error(f"Error processing {gene_name}: {e}")

    if slurm:
        logger.success(f"Successfully submitted {len(fasta_files)} SLURM jobs.")


@app.command(hidden=True)
def internal_plot(aln_file: Path, plot_path: Path, ic_plot_path: Path, ic_csv_path: Path, title: str):
    """Hidden command for SLURM jobs to plot alignment and information content."""
    try:
        # Plot alignment
        mv = MsaViz(aln_file, format="fasta", show_consensus=True)
        mv.savefig(plot_path, dpi=300)

        # Calculate Information Content
        ic_df = calculate_information_content(aln_file)
        if not ic_df.empty:
            ic_df.to_csv(ic_csv_path, index=False)
            
            # Plot Information Content
            plt.figure(figsize=(15, 5))
            plt.fill_between(ic_df["Position"], ic_df["IC"], color="skyblue", alpha=0.4)
            plt.plot(ic_df["Position"], ic_df["IC"], color="Slateblue", alpha=0.6)
            plt.ylim(0, 2.1)
            plt.xlabel("Position")
            plt.ylabel("Information Content (bits)")
            plt.title(f"Information Content for {title} (Schneider & Stephens Correction)")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(ic_plot_path, dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error in internal-plot for {aln_file}: {e}")
        sys.exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
