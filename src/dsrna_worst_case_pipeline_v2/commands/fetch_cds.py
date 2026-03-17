import json
import typer
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from loguru import logger
from dsrna_worst_case_pipeline_v2.utils.orthodb import find_cluster_for_gene, fetch_fasta

DEFAULT_TAXON = "6656"  # Arthropoda

def fetch_cds(
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input", "-i"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o"),
    species_file: Optional[Path] = typer.Option(Path("input/insects_list.csv"), "--species-list", "-s"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r"),
    taxon: str = typer.Option(DEFAULT_TAXON, "--taxon", "-t"),
):
    """Fetch CDS sequences and prepare folder structure."""
    if not input_file.exists():
        logger.error(f"Input file {input_file} not found.")
        raise typer.Exit(1)
    
    (output_dir / "orthologs").mkdir(parents=True, exist_ok=True)
    ref_base = output_dir / "Organisms" / reference_organism.replace(" ", "_")
    ref_base.mkdir(parents=True, exist_ok=True)
    
    target_species = set()
    if species_file and species_file.exists():
        df = pd.read_csv(species_file)
        target_species = set(df[df.columns[0]].astype(str).str.strip())
    target_species.add(reference_organism)
    
    lines = input_file.read_text().splitlines()
    rows = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith('#'): continue
        parts = [p.strip() for p in l.split(',')]
        if len(parts) >= 2:
            rows.append({"gene_id": parts[0], "gene_name": parts[1]})
        elif len(parts) == 1:
            rows.append({"gene_id": parts[0], "gene_name": parts[0]})
    df_genes = pd.DataFrame(rows)
    
    logger.info(f"Processing {len(df_genes)} genes for {reference_organism}...")
    for _, row in tqdm(df_genes.iterrows(), total=len(df_genes), desc="Fetching"):
        cluster_id = find_cluster_for_gene(row["gene_id"], row["gene_name"], taxon)
        if not cluster_id:
            logger.warning(f"No cluster found for gene {row['gene_name']} ({row['gene_id']})")
            continue
        fasta = fetch_fasta(cluster_id, taxon)
        if fasta:
            recs = [f">{p.strip()}" for p in fasta.split(">") if p.strip() and json.loads(p.split("\n", 1)[0][p.find("{"):p.rfind("}")+1]).get("organism_name", "").strip() in target_species]
            if recs:
                safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in row["gene_name"].replace(" ", "_"))
                (output_dir / "orthologs" / f"{safe_name}_{row['gene_id']}_{cluster_id}.fasta").write_text("\n".join(recs) + "\n")
