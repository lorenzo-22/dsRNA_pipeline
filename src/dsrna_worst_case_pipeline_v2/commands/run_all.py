import typer
from pathlib import Path
from typing import Optional
from loguru import logger

from dsrna_worst_case_pipeline_v2.commands.fetch_cds import fetch_cds, DEFAULT_TAXON
from dsrna_worst_case_pipeline_v2.commands.plot_lengths import plot_lengths
from dsrna_worst_case_pipeline_v2.commands.align_sequences import align
from dsrna_worst_case_pipeline_v2.commands.pairwise_align import pairwise
from dsrna_worst_case_pipeline_v2.commands.calculate_accessibility import accessibility
from dsrna_worst_case_pipeline_v2.commands.bowtie_match import bowtie

def run_all(
    input_file: Path = typer.Option(Path("input/gene_ids.txt"), "--input", "-i", help="Path to gene IDs file"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Base output directory"),
    species_file: Optional[Path] = typer.Option(Path("input/insects_list.csv"), "--species-list", "-s", help="Path to species list CSV"),
    reference_organism: str = typer.Option("Phaedon cochleariae", "--reference", "-r", help="Reference organism name"),
    taxon: str = typer.Option(DEFAULT_TAXON, "--taxon", "-t", help="Taxon ID for OrthoDB"),
    slurm: bool = typer.Option(True, help="Whether to use Slurm for compute-intensive steps"),
    mem: str = typer.Option("16G", help="Memory allocation for Slurm jobs"),
    skip_fetch: bool = typer.Option(False, "--skip-fetch", help="Skip the initial sequence fetching step"),
):
    """Run the pipeline steps: fetch, plot, MSA, bowtie, pairwise, and accessibility."""
    logger.info(f"Starting pipeline for reference: {reference_organism}")

    if not skip_fetch:
        logger.info("Step 1/6: Fetching CDS sequences...")
        fetch_cds(input_file, output_dir, species_file, reference_organism, taxon)
    else:
        logger.info("Step 1/6: Skipping fetch as requested.")

    fasta_dir = output_dir / "orthologs"
    if not fasta_dir.exists():
        logger.error(f"Fasta directory {fasta_dir} does not exist. Cannot proceed.")
        raise typer.Exit(1)

    logger.info("Step 2/6: Generating length distribution plots...")
    plot_lengths(fasta_dir, output_dir, reference_organism)

    logger.info("Step 3/6: Performing Multiple Sequence Alignment (MSA)...")
    align(fasta_dir, output_dir, reference_organism, input_file, slurm, mem)

    logger.info("Step 4/6: Finding 21-mer matches with Bowtie...")
    bowtie(fasta_dir, output_dir, reference_organism, input_file, slurm, mem)

    logger.info("Step 5/6: Performing Pairwise Alignments...")
    pairwise(fasta_dir, output_dir, reference_organism, input_file, slurm, mem)

    logger.info("Step 6/6: Calculating Accessibility and windowed identity...")
    accessibility(fasta_dir, output_dir, reference_organism, input_file, slurm, mem)

    logger.info("Pipeline execution finished. Run 'aggregate' separately after jobs complete.")
