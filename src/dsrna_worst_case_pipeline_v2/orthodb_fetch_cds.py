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
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pymsaviz import MsaViz

# Try to import ViennaRNA
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

app = typer.Typer(help="Fetch CDS sequences from OrthoDB and analyze them (length, MSA, pairwise alignment, accessibility).")

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
    if not needle_file.exists(): return stats
    content = needle_file.read_text()
    id_match = re.search(r"Identity:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    sim_match = re.search(r"Similarity:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    gap_match = re.search(r"Gaps:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    if id_match: stats["Identity"] = float(id_match.group(1))
    if sim_match: stats["Similarity"] = float(sim_match.group(1))
    if gap_match: stats["Gaps"] = float(gap_match.group(1))
    return stats


def get_anchored_sequences(needle_file: Path) -> Tuple[str, str]:
    """Extract aligned sequences from Needle file and remove reference gaps."""
    try:
        alignment = AlignIO.read(needle_file, "emboss")
        ref_seq, que_seq = str(alignment[0].seq), str(alignment[1].seq)
        anchored_ref, anchored_que = [], []
        for r, q in zip(ref_seq, que_seq):
            if r != '-':
                anchored_ref.append(r)
                anchored_que.append(q)
        return "".join(anchored_ref), "".join(anchored_que)
    except Exception as e:
        logger.error(f"Error parsing sequences from {needle_file}: {e}")
        return "", ""


def run_vienna_accessibility(sequence: str, window_size: int = 150, max_span: int = 100) -> np.ndarray:
    """Calculate per-nucleotide accessibility (u=1) using ViennaRNA."""
    if not HAS_VIENNA:
        logger.warning("ViennaRNA not found. Returning zero accessibility.")
        return np.zeros(len(sequence))
    try:
        seq_len = len(sequence)
        w = min(window_size, seq_len)
        l = min(max_span, seq_len)
        probs_matrix = RNA.pfl_fold_up(sequence, 1, w, l)
        acc = np.zeros(seq_len)
        for i in range(1, seq_len + 1):
            if i < len(probs_matrix) and 1 < len(probs_matrix[i]):
                acc[i-1] = probs_matrix[i][1]
        return acc
    except Exception as e:
        logger.error(f"Error calculating accessibility: {e}")
        return np.zeros(len(sequence))


def calculate_column_ic(column: List[str]) -> float:
    """Calculate Shannon IC for a single alignment column."""
    valid_bases = [b.upper() for b in column if b.upper() in 'ACGTU']
    n = len(valid_bases)
    if n == 0: return 0.0
    s, h_max, ln2 = 4, 2.0, math.log(2)
    counts = {b: valid_bases.count(b) for b in 'ACGT'}
    if 'U' in valid_bases: counts['T'] += valid_bases.count('U')
    h_l = 0
    for b in 'ACGT':
        p = counts[b] / n
        if p > 0: h_l -= p * math.log2(p)
    e_n = (s - 1) / (2 * n * ln2)
    return max(0, h_max - (h_l + e_n))


def calculate_information_content(alignment_file: Path, reference_id: Optional[str] = None) -> pd.DataFrame:
    """Calculate IC per alignment position, mapped to reference."""
    records = list(SeqIO.parse(alignment_file, "fasta"))
    if not records: return pd.DataFrame()
    aln_len = len(records[0].seq)
    ref_seq = next((str(r.seq) for r in records if reference_id == r.id), None)
    results = []
    ref_pos = 0
    for i in range(aln_len):
        col = [str(r.seq[i]) for r in records]
        is_ref_gap = False
        if ref_seq:
            if ref_seq[i] != '-': ref_pos += 1
            else: is_ref_gap = True
        results.append({
            "Position": i + 1, 
            "ReferencePosition": ref_pos if not is_ref_gap else None,
            "IC": calculate_column_ic(col)
        })
    return pd.DataFrame(results)


@app.command()
def fetch_cds(
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input", "-i"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o"),
    species_file: Optional[Path] = typer.Option(Path("input/insects_list.csv"), "--species-list", "-s"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    taxon: str = typer.Option(DEFAULT_TAXON, "--taxon", "-t"),
):
    """Fetch CDS sequences and prepare folder structure."""
    if not input_file.exists(): raise typer.Exit(1)
    (output_dir / "orthologs").mkdir(parents=True, exist_ok=True)
    ref_base = output_dir / "Organisms" / reference_organism.replace(" ", "_")
    ref_base.mkdir(parents=True, exist_ok=True)
    target_species = set()
    if species_file and species_file.exists():
        df = pd.read_csv(species_file)
        target_species = set(df[df.columns[0]].astype(str).str.strip())
    target_species.add(reference_organism)
    lines = input_file.read_text().splitlines()
    df_genes = pd.DataFrame([([s.strip() for s in l.split(",", 1)] if "," in l else [l.strip(), l.strip()]) for l in lines if l.strip()], columns=["gene_id", "gene_name"])
    logger.info(f"Processing {len(df_genes)} genes for {reference_organism}...")
    for _, row in tqdm(df_genes.iterrows(), total=len(df_genes), desc="Fetching"):
        cluster_id = find_cluster_for_gene(row["gene_id"], row["gene_name"], taxon)
        if not cluster_id: continue
        fasta = fetch_fasta(cluster_id, taxon)
        if fasta:
            recs = [f">{p.strip()}" for p in fasta.split(">") if p.strip() and json.loads(p.split("\n", 1)[0][p.find("{"):p.rfind("}")+1]).get("organism_name", "").strip() in target_species]
            if recs:
                safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in row["gene_name"].replace(" ", "_"))
                (output_dir / "orthologs" / f"{safe_name}_{row['gene_id']}_{cluster_id}.fasta").write_text("\n".join(recs) + "\n")


@app.command()
def plot_lengths(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
):
    """Generate CDS length distribution plots."""
    summary_dir = output_base / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    length_dist_dir = output_base / "length_distributions"
    length_dist_dir.mkdir(parents=True, exist_ok=True)
    all_data = []
    for f in list(fasta_dir.glob("*.fasta")):
        gene = get_gene_name(f.stem)
        for r in SeqIO.parse(f, "fasta"):
            try:
                meta = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1])
                all_data.append({"Gene": gene, "Organism": meta.get("organism_name", "Unknown"), "Length": len(r.seq)})
            except: pass
    if not all_data: return
    df = pd.DataFrame(all_data)
    unique_genes = df["Gene"].unique()
    logger.info(f"Generating plots for {len(unique_genes)} unique genes...")
    for gene in tqdm(unique_genes, desc="Generating gene plots"):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df["Gene"] == gene], x="Organism", y="Length")
        plt.xticks(rotation=45, ha='right'); plt.title(f"CDS Length Distribution: {gene}"); plt.tight_layout()
        plt.savefig(length_dist_dir / f"{gene}_length_distribution.png"); plt.close()
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.groupby(["Gene", "Organism"], observed=True)["Length"].mean().reset_index(), x="Gene", y="Length", hue="Organism")
    plt.xticks(rotation=45, ha='right'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(summary_dir / "summary_length_comparison.png"); plt.close()


@app.command()
def align_sequences(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G", help="Memory for SLURM job."),
):
    """Perform MSA using Clustal Omega."""
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="MSA"):
        gene = get_gene_name(f.stem)
        g_dir = output_base / "msa" / gene
        for d in ["fasta", "plots", "slurm"]: (g_dir / d).mkdir(parents=True, exist_ok=True)
        recs = list(SeqIO.parse(f, "fasta"))
        if len(recs) < 2: continue
        ref_idx = next((i for i, r in enumerate(recs) if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), -1)
        renamed, ref_id = [], None
        for i, r in enumerate(recs):
            try:
                name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown").replace(" ", "_")
            except: name = r.id
            rid = f"{name}_{i}"
            if i == ref_idx: ref_id = rid; renamed.insert(0, SeqRecord(r.seq, id=rid, description=""))
            else: renamed.append(SeqRecord(r.seq, id=rid, description=""))
        temp_in = g_dir / "fasta" / "renamed_orthologs.fasta"
        SeqIO.write(renamed, temp_in, "fasta")
        aln, plot, ic_p, ic_c = g_dir / "fasta" / "aligned.fasta", g_dir / "plots" / "alignment.png", g_dir / "plots" / "information_content.png", g_dir / "information_content.csv"
        if slurm:
            script = g_dir / "slurm" / f"msa_{f.stem}.sh"
            ref_arg = f'--reference-id "{ref_id}"' if ref_id else ""
            content = f"#!/bin/bash\n#SBATCH --job-name=msa_{gene[:10]}\n#SBATCH --output={g_dir}/slurm/job.out\n#SBATCH --cpus-per-task=4\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\nmodule load clustal-omega\nclustalo -i {temp_in.resolve()} -o {aln.resolve()} --force --outfmt=fasta --threads=4\n{sys.executable} {Path(__file__).resolve()} internal-msa-plot {aln.resolve()} {plot.resolve()} {ic_p.resolve()} {ic_c.resolve()} \"{gene}\" {ref_arg}\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            subprocess.run(["bash", "-c", f"module load clustal-omega && clustalo -i {temp_in} -o {aln} --force --outfmt=fasta"], check=True)
            internal_msa_plot(aln, plot, ic_p, ic_c, gene, ref_id)


@app.command()
def pairwise_align(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G", help="Memory for SLURM job."),
):
    """Perform Pairwise Alignment anchored to reference using EMBOSS Needle."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Pairwise"):
        gene = get_gene_name(f.stem)
        g_dir = org_base / gene / "pairwise_alignments"
        for d in ["fasta", "plots", "needle"]: (g_dir / d).mkdir(parents=True, exist_ok=True)
        recs = list(SeqIO.parse(f, "fasta"))
        ref_rec = next((r for r in recs if reference_organism.lower() in json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "").lower()), None)
        if not ref_rec: continue
        ref_tmp = g_dir / "fasta" / "reference.fasta"
        SeqIO.write(ref_rec, ref_tmp, "fasta")
        if slurm:
            script = g_dir / "slurm_pairwise.sh"
            content = f"#!/bin/bash\n#SBATCH --job-name=pair_{gene[:10]}\n#SBATCH --output={g_dir}/job_pairwise.out\n#SBATCH --mem={mem}\n#SBATCH --time=01:00:00\n\nmodule load EMBOSS\n{sys.executable} {Path(__file__).resolve()} internal-pairwise-run \"{f.resolve()}\" \"{ref_tmp.resolve()}\" \"{g_dir.resolve()}\" \"{reference_organism}\" \"{gene}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_pairwise_run(f, ref_tmp, g_dir, reference_organism, gene)


@app.command()
def calculate_accessibility(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    slurm: bool = typer.Option(True),
    mem: str = typer.Option("16G", help="Memory for SLURM job."),
):
    """Calculate reference-anchored windowed accessibility and conservation."""
    org_base = output_base / "Organisms" / reference_organism.replace(" ", "_")
    for f in tqdm(list(fasta_dir.glob("*.fasta")), desc="Accessibility"):
        gene = get_gene_name(f.stem)
        pw_dir = org_base / gene / "pairwise_alignments"
        acc_dir = org_base / gene / "accessibility"
        for d in ["plots", "data"]: (acc_dir / d).mkdir(parents=True, exist_ok=True)
        ref_tmp = pw_dir / "fasta" / "reference.fasta"
        if not ref_tmp.exists(): continue
        if slurm:
            script = acc_dir / "slurm_acc.sh"
            content = f"#!/bin/bash\n#SBATCH --job-name=acc_{gene[:10]}\n#SBATCH --output={acc_dir}/job_acc.out\n#SBATCH --mem={mem}\n#SBATCH --time=02:00:00\n\n{sys.executable} {Path(__file__).resolve()} internal-accessibility-run \"{f.resolve()}\" \"{ref_tmp.resolve()}\" \"{pw_dir.resolve()}\" \"{acc_dir.resolve()}\" \"{reference_organism}\" \"{gene}\"\n"
            script.write_text(content)
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            internal_accessibility_run(f, ref_tmp, pw_dir, acc_dir, reference_organism, gene)


@app.command(hidden=True)
def internal_pairwise_run(fasta_file: Path, ref_tmp: Path, g_dir: Path, reference_organism: str, gene_name: str):
    recs = list(SeqIO.parse(fasta_file, "fasta"))
    metrics, anchored_recs = [], []
    ref_rec = list(SeqIO.parse(ref_tmp, "fasta"))[0]
    anchored_recs.append(SeqRecord(ref_rec.seq, id=f"REF_{reference_organism.replace(' ', '_')}", description=""))
    for r in recs:
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        q_tmp = g_dir / "fasta" / f"temp_{org_name.replace(' ', '_')}.fasta"
        SeqIO.write(r, q_tmp, "fasta")
        out_n = g_dir / "needle" / f"{org_name.replace(' ', '_')}_vs_ref.needle"
        subprocess.run(["needle", "-asequence", str(ref_tmp), "-bsequence", str(q_tmp), "-outfile", str(out_n), "-datafile", "EDNAFULL", "-gapopen", 10, "-gapextend", 0.5], check=True, capture_output=True)
        metrics.append({**parse_needle_output(out_n), "Organism": org_name})
        ref_anch, que_anch = get_anchored_sequences(out_n)
        if que_anch: anchored_recs.append(SeqRecord(Seq(que_anch), id=org_name.replace(' ', '_'), description=""))
        q_tmp.unlink()
    if metrics:
        pd.DataFrame(metrics).to_csv(g_dir / "pairwise_metrics.csv", index=False)
        plt.figure(figsize=(10, 6)); sns.barplot(data=pd.DataFrame(metrics).melt(id_vars="Organism", value_vars=["Similarity", "Gaps"]), x="Organism", y="value", hue="variable")
        plt.xticks(rotation=45, ha='right'); plt.title(f"Pairwise Metrics: {gene_name}"); plt.ylabel("Percentage (%)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(g_dir / "plots" / "metrics_comparison.png", dpi=300); plt.close()
    if len(anchored_recs) > 1:
        a_fa = g_dir / "fasta" / "anchored_alignment.fasta"
        SeqIO.write(anchored_recs, a_fa, "fasta")
        try:
            mv = MsaViz(a_fa, format="fasta", show_consensus=True)
            fig = mv.plotfig(); fig.suptitle(f"Reference-Anchored Alignment: {gene_name}\n(Gaps in reference removed)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(g_dir / "plots" / "anchored_alignment.png", dpi=300); plt.close(fig)
        except Exception as e: print(f"Error plotting anchored: {e}")


@app.command(hidden=True)
def internal_accessibility_run(fasta_file: Path, ref_tmp: Path, pw_dir: Path, acc_dir: Path, reference_organism: str, gene_name: str):
    recs = list(SeqIO.parse(fasta_file, "fasta"))
    ref_rec = list(SeqIO.parse(ref_tmp, "fasta"))[0]
    ref_acc = run_vienna_accessibility(str(ref_rec.seq))
    window_data = []
    for r in recs:
        org_name = json.loads(r.description[r.description.find("{"):r.description.rfind("}")+1]).get("organism_name", "Unknown")
        if org_name.lower() == reference_organism.lower(): continue
        needle_file = pw_dir / "needle" / f"{org_name.replace(' ', '_')}_vs_ref.needle"
        if not needle_file.exists(): continue
        aln = AlignIO.read(needle_file, "emboss")
        ref_aln, que_aln = str(aln[0].seq), str(aln[1].seq)
        que_acc = run_vienna_accessibility(str(r.seq))
        ref_to_col = [i for i, b in enumerate(ref_aln) if b != '-']
        que_to_col = [i for i, b in enumerate(que_aln) if b != '-']
        col_to_que_acc = {col: que_acc[q_idx] for q_idx, col in enumerate(que_to_col)}
        results = []
        for start_ref in range(len(ref_to_col) - 299):
            end_ref = start_ref + 299
            start_col, end_col = ref_to_col[start_ref], ref_to_col[end_ref]
            span = end_col - start_col + 1
            win_ref_aln, win_que_aln = ref_aln[start_col:end_col+1], que_aln[start_col:end_col+1]
            ident = sum(1 for a, b in zip(win_ref_aln, win_que_aln) if a == b and a != '-') / span
            win_ic = np.mean([calculate_column_ic([ref_aln[c], que_aln[c]]) for c in range(start_col, end_col+1)])
            win_q_acc = sum(col_to_que_acc.get(c, 0.0) for c in range(start_col, end_col+1)) / span
            results.append({"RefPos": start_ref + 1, "Identity": ident, "IC": win_ic, "NTO_Acc": win_q_acc})
        window_data.append(pd.DataFrame(results))
    if window_data:
        avg_df = pd.concat(window_data).groupby("RefPos").mean().reset_index()
        avg_df.to_csv(acc_dir / "data" / "windowed_analysis.csv", index=False)
        plt.figure(figsize=(12, 5)); ref_win_acc = [np.mean(ref_acc[i:i+300]) for i in range(len(ref_acc)-299)]
        plt.plot(range(1, len(ref_win_acc)+1), ref_win_acc, label=f"Ref ({reference_organism})", color="black", lw=2)
        plt.plot(avg_df["RefPos"], avg_df["NTO_Acc"], label="Average NTOs", color="red", alpha=0.7)
        plt.title(f"Windowed Accessibility (300nt): {gene_name}"); plt.xlabel("Reference Nucleotide Position"); plt.ylabel("Probability Unpaired")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(acc_dir / "plots" / "windowed_accessibility.png", dpi=300); plt.close()
        plt.figure(figsize=(12, 5)); plt.plot(avg_df["RefPos"], avg_df["Identity"]*100, label="% Identity", color="blue")
        plt.plot(avg_df["RefPos"], avg_df["IC"]*50, label="IC (bits * 50)", color="green", alpha=0.6)
        plt.title(f"Windowed Conservation (300nt): {gene_name}"); plt.xlabel("Reference Nucleotide Position"); plt.ylabel("Score")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(acc_dir / "plots" / "windowed_conservation.png", dpi=300); plt.close()


@app.command(hidden=True)
def internal_msa_plot(aln_file: Path, plot_path: Path, ic_plot: Path, ic_csv: Path, title: str, reference_id: Optional[str] = None):
    try:
        mv = MsaViz(aln_file, format="fasta", show_consensus=True)
        fig = mv.plotfig(); fig.suptitle(f"Multiple Sequence Alignment: {title}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(plot_path, dpi=300); plt.close(fig)
        ic_df = calculate_information_content(aln_file, reference_id)
        if not ic_df.empty:
            ic_df.to_csv(ic_csv, index=False); plt.figure(figsize=(15, 5))
            plt.fill_between(ic_df["Position"], ic_df["IC"], color="skyblue", alpha=0.4); plt.plot(ic_df["Position"], ic_df["IC"], color="Slateblue", alpha=0.6)
            plt.ylim(0, 2.1); ref_name = reference_id.rsplit('_', 1)[0].replace('_', ' ') if reference_id else ""
            plt.xlabel(f"Alignment Position{' (Ref: ' + ref_name + ')' if ref_name else ''}"); plt.ylabel("Information Content (bits)"); plt.title(f"Information Content: {title}")
            plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(ic_plot, dpi=300); plt.close()
    except Exception as e: print(f"Error: {e}"); sys.exit(1)

def main(): app()
if __name__ == "__main__": main()
