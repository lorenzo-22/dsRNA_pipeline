import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from Bio import SeqIO
from dsrna_worst_case_pipeline_v2.utils.bio import get_gene_name

def plot_lengths(
    fasta_dir: Path = typer.Option(Path("output/orthologs"), "--input", "-i"),
    output_base: Path = typer.Option(Path("output"), "--output", "-o"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
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
            except Exception as e:
                logger.debug(f"Error parsing metadata for {r.id} in {f}: {e}")
                pass
                
    if not all_data:
        logger.warning("No data found for plotting.")
        return
        
    df = pd.DataFrame(all_data)
    unique_genes = df["Gene"].unique()
    logger.info(f"Generating plots for {len(unique_genes)} unique genes...")
    
    for gene in tqdm(unique_genes, desc="Generating gene plots"):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df["Gene"] == gene], x="Organism", y="Length")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"CDS Length Distribution: {gene}")
        plt.tight_layout()
        plt.savefig(length_dist_dir / f"{gene}_length_distribution.png")
        plt.close()
        
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.groupby(["Gene", "Organism"], observed=True)["Length"].mean().reset_index(), x="Gene", y="Length", hue="Organism")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(summary_dir / "summary_length_comparison.png")
    plt.close()
