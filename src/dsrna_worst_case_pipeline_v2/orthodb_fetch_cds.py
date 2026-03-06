import sys
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer(help="Fetch CDS sequences from OrthoDB for orthologs in a taxonomic group.")

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
    # Note: OrthoDB uses 'id' for the cluster ID in the fasta endpoint
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
    # Try searching by gene ID first (using 'query' parameter)
    logger.debug(f"Searching for gene ID: {gene_id}")
    search_data = fetch_orthodb_data("search", {"query": gene_id, "level": taxon})
    
    if not search_data or not search_data.get("data"):
        # If not found, try searching by gene name
        logger.debug(f"Gene ID {gene_id} not found, searching for gene name: {gene_name}")
        search_data = fetch_orthodb_data("search", {"query": gene_name, "level": taxon})

    if not search_data or not search_data.get("data"):
        # Last resort: search without level constraint
        logger.debug(f"Searching for gene name: {gene_name} without level constraint")
        search_data = fetch_orthodb_data("search", {"query": gene_name})

    if not search_data or not search_data.get("data"):
        return None

    # Pick the most relevant cluster that has the taxon in its ID if possible
    # or just pick the first one from bigdata/data.
    
    if search_data.get("bigdata"):
        for entry in search_data["bigdata"]:
            cluster_id = entry.get("id")
            if cluster_id and f"at{taxon}" in cluster_id:
                return cluster_id
        # If none matched specifically, pick the first from bigdata
        return search_data["bigdata"][0].get("id")

    # Fallback to the first ID in 'data'
    return search_data["data"][0]


@app.command()
def fetch_cds(
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input", "-i", help="Input file with gene IDs and names (CSV/TXT)."),
    output_dir: Path = typer.Option(Path("output/orthologs"), "--output", "-o", help="Output directory for FASTA files."),
    taxon: str = typer.Option(DEFAULT_TAXON, "--taxon", "-t", help="NCBI Taxonomy ID for filtering (default: 6656 for Arthropoda)."),
):
    """
    Main entry point to fetch CDS sequences for a list of genes.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the input file.
    try:
        # Load all columns, we only care about the first two.
        df = pd.read_csv(input_file, header=None)
        if df.shape[1] >= 2:
            df.columns = ["gene_id", "gene_name"] + list(df.columns[2:])
        else:
            df.columns = ["gene_id"]
            df["gene_name"] = df["gene_id"]
            
        df["gene_id"] = df["gene_id"].astype(str).str.strip()
        df["gene_name"] = df["gene_name"].astype(str).str.strip()
    except Exception as e:
        logger.error(f"Error reading input file with pandas: {e}")
        # Fallback to simple line reading
        lines = input_file.read_text().splitlines()
        data = []
        for line in lines:
            if "," in line:
                data.append([s.strip() for s in line.split(",", 1)])
            else:
                data.append([line.strip(), line.strip()])
        df = pd.DataFrame(data, columns=["gene_id", "gene_name"])

    logger.info(f"Processing {len(df)} genes...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching sequences"):
        gene_id = row["gene_id"]
        gene_name = row["gene_name"]
        
        logger.info(f"Processing {gene_id} ({gene_name})")
        cluster_id = find_cluster_for_gene(gene_id, gene_name, taxon)
        
        if not cluster_id:
            logger.warning(f"Could not find cluster for {gene_id} ({gene_name})")
            continue
            
        logger.info(f"Found cluster {cluster_id}. Fetching FASTA...")
        
        fasta_content = fetch_fasta(cluster_id, taxon)
        
        if fasta_content and fasta_content.strip():
            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in gene_name.replace(" ", "_"))
            output_file = output_dir / f"{safe_name}_{gene_id}_{cluster_id}.fasta"
            output_file.write_text(fasta_content)
            logger.info(f"Saved {len(fasta_content)} bytes to {output_file}")
        else:
            logger.warning(f"No FASTA content found for cluster {cluster_id}")


def main():
    app()


if __name__ == "__main__":
    main()
