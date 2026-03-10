import json
import sys
import subprocess
import os
import math
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import httpx
import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import seaborn as sns
from pymsaviz import MsaViz

app = typer.Typer(help="Fetch CDS sequences from OrthoDB and analyze them (length, MSA, reference-anchored pairwise alignment).")

# Configure loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{message}</cyan>", level="INFO")

ORTHODB_BASE_URL = "https://data.orthodb.org/current"
DEFAULT_TAXON = "6656"  # Arthropoda


def get_gene_name(fasta_stem: str) -> str:
    """Consistently extract gene name from filename stem."""
    parts = fasta_stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else fasta_stem


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
    params = {"id": cluster_id, "species": taxon, "seqtype": seq_type}
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error(f"Error fetching FASTA for cluster {cluster_id}: {e}")
        return None


def find_cluster_for_gene(gene_id: str, gene_name: str, taxon: str) -> Optional[str]:
    """Find the OrthoDB cluster ID for a given gene ID or name."""
    search_data = fetch_orthodb_data("search", {"query": gene_id, "level": taxon})
    if not search_data or not search_data.get("data"):
        search_data = fetch_orthodb_data("search", {"query": gene_name, "level": taxon})
    if not search_data or not search_data.get("data"):
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


def parse_needle_output(needle_file: Path) -> Dict:
    """Parse similarity, identity and gaps from EMBOSS Needle output."""
    stats = {"Identity": 0.0, "Similarity": 0.0, "Gaps": 0.0}
    if not needle_file.exists():
        return stats
    
    content = needle_file.read_text()
    # Regex to find percentages in EMBOSS format: (75.0%)
    id_match = re.search(r"Identity:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    sim_match = re.search(r"Similarity:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    gap_match = re.search(r"Gaps:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    
    if id_match: stats["Identity"] = float(id_match.group(1))
    if sim_match: stats["Similarity"] = float(sim_match.group(1))
    if gap_match: stats["Gaps"] = float(gap_match.group(1))
    
    return stats


def get_anchored_sequences(needle_file: Path) -> Tuple[str, str]:
    """
    Extract the aligned sequences from Needle file and remove 
    positions where the reference has a gap.
    Returns (anchored_ref, anchored_query)
    """
    if not needle_file.exists():
        return "", ""
    
    ref_aln = []
    que_aln = []
    
    # Simple EMBOSS Needle sequence parser
    # Looks for lines starting with sequence names
    with open(needle_file, 'r') as f:
        lines = f.readlines()
        
    # EMBOSS output is tricky to parse manually for sequences, 
    # but we can look for the block lines
    # Usually: name position seq position
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        # Extract sequence lines - this is a bit heuristic for standard Needle output
        parts = line.split()
        if len(parts) >= 3 and not parts[0].isdigit():
            # We assume first sequence encountered is Ref, second is Query
            # This is fragile, but since we controlled the input to needle, 
            # we know the order.
            pass

    # A more robust way is to use Bio.AlignIO if Needle format is supported
    from Bio import AlignIO
    try:
        alignment = AlignIO.read(needle_file, "emboss")
        ref_seq = str(alignment[0].seq)
        que_seq = str(alignment[1].seq)
        
        anchored_ref = []
        anchored_que = []
        
        for r, q in zip(ref_seq, que_seq):
            if r != '-':
                anchored_ref.append(r)
                anchored_que.append(q)
        
        return "".join(anchored_ref), "".join(anchored_que)
    except Exception as e:
        logger.error(f"Error parsing sequences from {needle_file}: {e}")
        return "", ""


@app.command()
def fetch_cds(
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input", "-i", help="Input file with gene IDs and names."),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Base output directory."),
    species_file: Optional[Path] = typer.Option(Path("input/insects_list.csv"), "--species-list", "-s", help="Optional CSV file with species to keep."),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r", help="Designated reference organism."),
    taxon: str = typer.Option(DEFAULT_TAXON, "--taxon", "-t", help="NCBI Taxonomy ID for filtering."),
):
    """
    Fetch CDS sequences and prepare reference organism folder.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise typer.Exit(code=1)

    orthologs_dir = output_dir / "orthologs"
    orthologs_dir.mkdir(parents=True, exist_ok=True)

    # Only create folder for reference organism
    ref_safe = reference_organism.replace(" ", "_")
    organisms_base = output_dir / "Organisms"
    ref_base = organisms_base / ref_safe
    ref_base.mkdir(parents=True, exist_ok=True)

    target_species = set()
    if species_file and species_file.exists():
        species_df = pd.read_csv(species_file)
        col = "Species" if "Species" in species_df.columns else species_df.columns[0]
        target_species = set(species_df[col].astype(str).str.strip().tolist())
    
    target_species.add(reference_organism)

    data = []
    lines = input_file.read_text().splitlines()
    for line in lines:
        if not line.strip(): continue
        parts = [s.strip() for s in line.split(",", 1)]
        data.append(parts if len(parts) == 2 else [parts[0], parts[0]])
    df = pd.DataFrame(data, columns=["gene_id", "gene_name"])

    logger.info(f"Processing {len(df)} genes for reference {reference_organism}...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching sequences"):
        gene_id, gene_name = row["gene_id"], row["gene_name"]
        cluster_id = find_cluster_for_gene(gene_id, gene_name, taxon)
        if not cluster_id: continue
        fasta_content = fetch_fasta(cluster_id, taxon)
        if fasta_content and fasta_content.strip():
            filtered_records = []
            for part in fasta_content.split(">"):
                if not part.strip(): continue
                header = part.split("\n", 1)[0]
                try:
                    meta = json.loads(header[header.find("{") : header.rfind("}") + 1])
                    if meta.get("organism_name", "").strip() in target_species:
                        filtered_records.append(f">{part.strip()}")
                except: pass
            
            fasta_content = "\n".join(filtered_records) + "\n" if filtered_records else ""
            if fasta_content.strip():
                safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in gene_name.replace(" ", "_"))
                output_file = orthologs_dir / f"{safe_name}_{gene_id}_{cluster_id}.fasta"
                output_file.write_text(fasta_content)


@app.command()
def plot_lengths(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i", help="Directory containing FASTAs."),
    output_base: Path = typer.Option(Path("output"), "--output", "-o", help="Base output directory."),
):
    """Generate CDS length distribution plots."""
    if not fasta_dir.exists(): raise typer.Exit(code=1)
    
    msa_base = output_base / "msa"
    summary_dir = output_base / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    for fasta_file in tqdm(list(fasta_dir.glob("*.fasta")), desc="Parsing FASTAs"):
        gene_name = get_gene_name(fasta_file.stem)
        for record in SeqIO.parse(fasta_file, "fasta"):
            try:
                meta = json.loads(record.description[record.description.find("{") : record.description.rfind("}") + 1])
                all_data.append({"Gene": gene_name, "Organism": meta.get("organism_name", "Unknown"), "Length": len(record.seq)})
            except: pass
    
    if not all_data: return
    df = pd.DataFrame(all_data)
    
    for gene in df["Gene"].unique():
        gene_dir = msa_base / gene
        gene_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df["Gene"] == gene], x="Organism", y="Length")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"CDS Length Distribution: {gene}")
        plt.tight_layout()
        plt.savefig(gene_dir / "length_distribution.png")
        plt.close()

    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.groupby(["Gene", "Organism"], observed=True)["Length"].mean().reset_index(), x="Gene", y="Length", hue="Organism")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(summary_dir / "summary_length_comparison.png")
    plt.close()


@app.command()
def align_sequences(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
):
    """Perform MSA using Clustal Omega."""
    msa_base = output_base / "msa"
    for fasta_file in tqdm(list(fasta_dir.glob("*.fasta")), desc="MSA"):
        gene_name = get_gene_name(fasta_file.stem)
        gene_dir = msa_base / gene_name
        gene_dir.mkdir(parents=True, exist_ok=True)
        records = list(SeqIO.parse(fasta_file, "fasta"))
        if len(records) < 2: continue
        
        # Find reference index
        ref_idx = -1
        max_len = -1
        for i, r in enumerate(records):
            try:
                meta = json.loads(r.description[r.description.find("{") : r.description.rfind("}") + 1])
                if reference_organism.lower() in meta.get("organism_name", "").lower():
                    if len(r.seq) > max_len:
                        max_len = len(r.seq)
                        ref_idx = i
            except: pass
        
        renamed = []
        ref_id = None
        for i, r in enumerate(records):
            try:
                meta = json.loads(r.description[r.description.find("{") : r.description.rfind("}") + 1])
                name = meta.get("organism_name", "Unknown").replace(" ", "_")
            except: name = r.id
            rid = f"{name}_{i}"
            if i == ref_idx:
                ref_id = rid
                renamed.insert(0, SeqRecord(r.seq, id=rid, description=""))
            else:
                renamed.append(SeqRecord(r.seq, id=rid, description=""))
        
        temp_in = gene_dir / "renamed_orthologs.fasta"
        SeqIO.write(renamed, temp_in, "fasta")
        
        aln = gene_dir / "aligned.fasta"
        plot = gene_dir / "alignment.png"
        ic_p = gene_dir / "information_content.png"
        ic_c = gene_dir / "information_content.csv"
        
        if slurm:
            script = gene_dir / "slurm" / f"msa_{fasta_file.stem}.sh"
            script.parent.mkdir(parents=True, exist_ok=True)
            ref_arg = f'--reference-id "{ref_id}"' if ref_id else ""
            content = f"#!/bin/bash\n#SBATCH --job-name=msa_{gene_name[:10]}\n#SBATCH --output={script.parent}/job.out\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=8G\n#SBATCH --time=02:00:00\n\nmodule load clustal-omega\nclustalo -i {temp_in.resolve()} -o {aln.resolve()} --force --outfmt=fasta --threads=4\n{sys.executable} {Path(__file__).resolve()} internal-plot {aln.resolve()} {plot.resolve()} {ic_p.resolve()} {ic_c.resolve()} \"{gene_name}\" {ref_arg}\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            subprocess.run(["bash", "-c", f"module load clustal-omega && clustalo -i {temp_in} -o {aln} --force --outfmt=fasta"], check=True)
            from dsrna_worst_case_pipeline_v2.orthodb_fetch_cds import internal_plot
            app.command(hidden=True)(internal_plot)(aln, plot, ic_p, ic_c, gene_name, ref_id)


@app.command()
def pairwise_align(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
):
    """
    Perform Pairwise Alignment anchored to reference and generate plots.
    """
    ref_safe = reference_organism.replace(" ", "_")
    org_base = output_base / "Organisms" / ref_safe
    
    for fasta_file in tqdm(list(fasta_dir.glob("*.fasta")), desc="Pairwise"):
        gene_name = get_gene_name(fasta_file.stem)
        gene_dir = org_base / gene_name
        gene_dir.mkdir(parents=True, exist_ok=True)
        
        # Identify reference record
        records = list(SeqIO.parse(fasta_file, "fasta"))
        ref_rec = None
        for r in records:
            try:
                meta = json.loads(r.description[r.description.find("{") : r.description.rfind("}") + 1])
                if reference_organism.lower() in meta.get("organism_name", "").lower():
                    if ref_rec is None or len(r.seq) > len(ref_rec.seq):
                        ref_rec = r
            except: pass
        
        if not ref_rec: continue
        
        ref_tmp = gene_dir / "reference.fasta"
        SeqIO.write(ref_rec, ref_tmp, "fasta")
        
        if slurm:
            script = gene_dir / "slurm_pairwise.sh"
            script.parent.mkdir(parents=True, exist_ok=True)
            content = f"""#!/bin/bash
#SBATCH --job-name=pair_{gene_name[:10]}
#SBATCH --output={gene_dir}/job_pairwise.out
#SBATCH --mem=4G
#SBATCH --time=01:00:00

module load EMBOSS
{sys.executable} {Path(__file__).resolve()} internal-pairwise-run "{fasta_file.resolve()}" "{ref_tmp.resolve()}" "{gene_dir.resolve()}" "{reference_organism}" "{gene_name}"
"""
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_pairwise_run(fasta_file, ref_tmp, gene_dir, reference_organism, gene_name)


@app.command(hidden=True)
def internal_pairwise_run(fasta_file: Path, ref_tmp: Path, gene_dir: Path, reference_organism: str, gene_name: str):
    """Helper to run pairwise alignments and generate anchored/metrics plots."""
    records = list(SeqIO.parse(fasta_file, "fasta"))
    metrics = []
    anchored_records = []
    
    # We need the reference record again
    ref_rec = list(SeqIO.parse(ref_tmp, "fasta"))[0]
    # Add reference itself to anchored records (first)
    anchored_records.append(SeqRecord(ref_rec.seq, id=f"{reference_organism.replace(' ', '_')}_REF", description=""))

    for r in records:
        try:
            meta = json.loads(r.description[r.description.find("{") : r.description.rfind("}") + 1])
            org_name = meta.get("organism_name", "Unknown")
        except: org_name = r.id
        
        if org_name.lower() == reference_organism.lower():
            continue
            
        que_tmp = gene_dir / f"temp_query_{org_name.replace(' ', '_')}.fasta"
        SeqIO.write(r, que_tmp, "fasta")
        
        out_needle = gene_dir / f"{org_name.replace(' ', '_')}_vs_ref.needle"
        subprocess.run([
            "needle", "-asequence", str(ref_tmp), "-bsequence", str(que_tmp),
            "-outfile", str(out_needle), "-datafile", "EDNAFULL", "-gapopen", "10", "-gapextend", "0.5"
        ], check=True, capture_output=True)
        
        # Parse Stats
        stats = parse_needle_output(out_needle)
        stats["Organism"] = org_name
        metrics.append(stats)
        
        # Anchored Sequence (no gaps in ref)
        ref_anchored, que_anchored = get_anchored_sequences(out_needle)
        if que_anchored:
            anchored_records.append(SeqRecord(Seq(que_anchored), id=org_name.replace(' ', '_'), description=""))
        
        que_tmp.unlink()

    # Save Metrics
    if metrics:
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(gene_dir / "pairwise_metrics.csv", index=False)
        
        # 1. Metrics Comparison Plot
        plt.figure(figsize=(10, 6))
        df_melt = df_metrics.melt(id_vars="Organism", value_vars=["Similarity", "Gaps"])
        sns.barplot(data=df_melt, x="Organism", y="value", hue="variable")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Pairwise Metrics against Reference: {gene_name}")
        plt.ylabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(gene_dir / "metrics_comparison.png", dpi=300)
        plt.close()

    # 2. Anchored Alignment Plot
    if len(anchored_records) > 1:
        anchored_fasta = gene_dir / "anchored_alignment.fasta"
        SeqIO.write(anchored_records, anchored_fasta, "fasta")
        try:
            mv = MsaViz(anchored_fasta, format="fasta", show_consensus=True)
            fig = mv.plotfig()
            fig.suptitle(f"Reference-Anchored Alignment: {gene_name}\n(Gaps in reference removed)", fontsize=14)
            fig.savefig(gene_dir / "anchored_alignment.png", dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Error plotting anchored alignment: {e}")


@app.command(hidden=True)
def internal_plot(aln_file: Path, plot_path: Path, ic_plot: Path, ic_csv: Path, title: str, reference_id: Optional[str] = None):
    try:
        mv = MsaViz(aln_file, format="fasta", show_consensus=True)
        fig = mv.plotfig()
        fig.suptitle(f"Multiple Sequence Alignment: {title}", fontsize=16)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        ic_df = calculate_information_content(aln_file, reference_id)
        if not ic_df.empty:
            ic_df.to_csv(ic_csv, index=False)
            plt.figure(figsize=(15, 5))
            plt.fill_between(ic_df["Position"], ic_df["IC"], color="skyblue", alpha=0.4)
            plt.plot(ic_df["Position"], ic_df["IC"], color="Slateblue", alpha=0.6)
            plt.ylim(0, 2.1); plt.xlabel(f"Alignment Position{' (Ref: ' + reference_id + ')' if reference_id else ''}")
            plt.ylabel("Information Content (bits)"); plt.title(f"Information Content: {title}")
            plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(ic_plot, dpi=300); plt.close()
    except Exception as e:
        print(f"Error in plotting: {e}"); sys.exit(1)

def main(): app()
if __name__ == "__main__": main()
