import typer
from dsrna_worst_case_pipeline_v2.commands.fetch_cds import fetch_cds
from dsrna_worst_case_pipeline_v2.commands.plot_lengths import plot_lengths
from dsrna_worst_case_pipeline_v2.commands.align_sequences import align_sequences, internal_msa_plot
from dsrna_worst_case_pipeline_v2.commands.pairwise_align import pairwise_align, internal_pairwise_run
from dsrna_worst_case_pipeline_v2.commands.calculate_accessibility import calculate_accessibility, internal_accessibility_run
from dsrna_worst_case_pipeline_v2.commands.bowtie_match import bowtie_match, internal_bowtie_run
from dsrna_worst_case_pipeline_v2.commands.aggregate_metrics import aggregate
from dsrna_worst_case_pipeline_v2.commands.run_all import run_all

app = typer.Typer(help="dsRNA Worst Case Pipeline v2")

# High-level command
app.command(name="run-all")(run_all)

# Main Step Commands
app.command(name="fetch")(fetch_cds)
app.command(name="plot-lengths")(plot_lengths)
app.command(name="align")(align_sequences)
app.command(name="pairwise")(pairwise_align)
app.command(name="accessibility")(calculate_accessibility)
app.command(name="bowtie")(bowtie_match)
app.command(name="aggregate")(aggregate)

# Hidden internal commands (used for SLURM/background tasks)
app.command(name="internal-msa-plot", hidden=True)(internal_msa_plot)
app.command(name="internal-pairwise-run", hidden=True)(internal_pairwise_run)
app.command(name="internal-accessibility-run", hidden=True)(internal_accessibility_run)
app.command(name="internal-bowtie-run", hidden=True)(internal_bowtie_run)

if __name__ == "__main__":
    app()
