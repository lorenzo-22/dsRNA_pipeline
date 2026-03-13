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

# Individual step commands
app.command(name="fetch")(fetch_cds)
app.command(name="plot-lengths")(plot_lengths)
app.command(name="aggregate")(aggregate)

# Alignment related
align_app = typer.Typer(help="Perform MSA")
align_app.command(name="run")(align_sequences)
align_app.command(name="internal-msa-plot", hidden=True)(internal_msa_plot)
app.add_typer(align_app, name="align")

# Pairwise related
pairwise_app = typer.Typer(help="Perform Pairwise Alignment")
pairwise_app.command(name="run")(pairwise_align)
pairwise_app.command(name="internal-pairwise-run", hidden=True)(internal_pairwise_run)
app.add_typer(pairwise_app, name="pairwise")

# Accessibility related
acc_app = typer.Typer(help="Calculate Accessibility")
acc_app.command(name="run")(calculate_accessibility)
acc_app.command(name="internal-accessibility-run", hidden=True)(internal_accessibility_run)
app.add_typer(acc_app, name="accessibility")

# Bowtie related
bowtie_app = typer.Typer(help="Find 21-mer matches using Bowtie")
bowtie_app.command(name="run")(bowtie_match)
bowtie_app.command(name="internal-bowtie-run", hidden=True)(internal_bowtie_run)
app.add_typer(bowtie_app, name="bowtie")

if __name__ == "__main__":
    app()
