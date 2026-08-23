"""Face recognition subcommands for the PhotoHeaven CLI."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from photoheaven.adapters.face.insightface import InsightFaceAnalyzer
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.face_assignment_service import (
    FaceAssignmentService,
)
from photoheaven.application.face_clustering_service import FaceClusteringService
from photoheaven.application.face_detection_service import FaceDetectionService
from photoheaven.application.face_identity_service import (
    ClusterSummary,
    FaceIdentityService,
    IdentitySummary,
)

logger = logging.getLogger(__name__)

faces_app = typer.Typer(
    name="faces",
    help="Detect, cluster, and name faces in your library.",
    no_args_is_help=True,
)
console = Console()


def _get_db_path(db_path: str | None) -> str:
    if db_path:
        resolved = db_path
    else:
        resolved = str(Path.cwd() / "db" / "photoheaven.db")
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    return resolved


@faces_app.command()
def detect(
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-analyse media even if already processed."
    ),
    det_size: int = typer.Option(
        640, "--det-size", help="InsightFace detector input size (e.g. 320 or 640)."
    ),
    min_confidence: float = typer.Option(
        0.0,
        "--min-confidence",
        help="Minimum face detection confidence to keep (0.0 keeps all).",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        help="Number of media files to fetch per database query.",
    ),
) -> None:
    """Run face detection on unprocessed images in the library."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)

    try:
        analyzer = InsightFaceAnalyzer(det_size=(det_size, det_size))
    except RuntimeError as exc:
        console.print(f"[red]Failed to load face analysis model:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error initialising face analyser")
        console.print(
            "[red]Unexpected error loading face analysis model:[/red] "
            f"{exc}"
        )
        raise typer.Exit(1) from exc

    service = FaceDetectionService(analyzer=analyzer, repository=repository)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # The service reports progress internally via logging; the spinner
        # simply shows that work is happening.
        task = progress.add_task("Detecting faces...", total=None)
        result = service.detect(
            force=force,
            min_confidence=min_confidence,
            batch_size=batch_size,
        )
        progress.remove_task(task)

    table = Table(title="Face detection summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Media processed", str(result.processed))
    table.add_row("Media skipped", str(result.skipped))
    table.add_row("Errors", str(result.errors))
    table.add_row("Faces detected", str(result.faces_detected))
    console.print(table)

    if result.errors:
        raise typer.Exit(1)


@faces_app.command()
def cluster(
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    eps: float = typer.Option(
        0.4,
        "--eps",
        help="Maximum cosine distance between two faces to be considered the same person.",
    ),
    min_samples: int = typer.Option(
        2,
        "--min-samples",
        help="Minimum faces required to form a cluster.",
    ),
    algorithm: str = typer.Option(
        "dbscan",
        "--algorithm",
        help="Clustering algorithm. Only dbscan is supported.",
    ),
) -> None:
    """Cluster detected faces into identity groups."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)
    service = FaceClusteringService(repository=repository)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Clustering faces...", total=None)
        try:
            result = service.cluster(eps=eps, min_samples=min_samples, algorithm=algorithm)
        except ValueError as exc:
            progress.remove_task(task)
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            progress.remove_task(task)
            logger.exception("Face clustering failed")
            console.print(f"[red]Face clustering failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.remove_task(task)

    table = Table(title="Face clustering summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Total faces", str(result.total_faces))
    table.add_row("Clusters", str(result.num_clusters))
    table.add_row("Clustered faces", str(result.clustered_faces))
    table.add_row("Noise / unclustered", str(result.noise_faces))
    console.print(table)


@faces_app.command()
def assign(
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    eps: float = typer.Option(
        0.4,
        "--eps",
        help="Maximum cosine distance to a known identity centroid for assignment.",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        help="Number of unassigned faces to fetch per query.",
    ),
) -> None:
    """Assign unlabelled faces to known identities using centroid distance."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)
    service = FaceAssignmentService(repository=repository)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Assigning faces to identities...", total=None)
        try:
            result = service.assign(eps=eps, batch_size=batch_size)
        except Exception as exc:
            progress.remove_task(task)
            logger.exception("Face assignment failed")
            console.print(f"[red]Face assignment failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.remove_task(task)

    table = Table(title="Incremental identity assignment summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Named identities used", str(result.identities_used))
    table.add_row("Faces assigned", str(result.assigned_faces))
    table.add_row("Faces left unassigned", str(result.unassigned_faces))
    console.print(table)


def _format_cluster_row(summary: ClusterSummary) -> tuple[str, ...]:
    identity = summary.identity_name or "—"
    sample = summary.sample_path or "—"
    return (
        str(summary.cluster_label),
        str(summary.photo_count),
        identity,
        sample,
    )


def _format_identity_row(summary: IdentitySummary) -> tuple[str, ...]:
    sample = summary.sample_path or "—"
    return (
        summary.identity_name,
        str(summary.photo_count),
        str(summary.face_count),
        sample,
    )


@faces_app.command("list")
def list_clusters(
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
    limit: int = typer.Option(
        100, "--limit", "-l", help="Maximum number of rows to show."
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Skip this many rows."
    ),
    by_identity: bool = typer.Option(
        False,
        "--by-identity",
        "-i",
        help="Group results by persistent identity instead of cluster label.",
    ),
) -> None:
    """List face clusters or identities ordered by size."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)
    service = FaceIdentityService(repository=repository)

    if by_identity:
        identities = service.list_identities(limit=limit, offset=offset)
        if not identities:
            console.print(
                "[yellow]No named identities found. "
                "Run 'ph faces name' or 'ph faces assign' first.[/yellow]"
            )
            raise typer.Exit(0)

        table = Table(title=f"Identities (showing {len(identities)})")
        table.add_column("Identity", style="green")
        table.add_column("Photos", style="magenta", justify="right")
        table.add_column("Faces", style="cyan", justify="right")
        table.add_column("Sample file", style="blue", no_wrap=True)

        for summary in identities:
            table.add_row(*_format_identity_row(summary))

        console.print(table)
        return

    clusters = service.list_clusters(limit=limit, offset=offset)

    if not clusters:
        console.print("[yellow]No clusters found. Run 'ph faces cluster' first.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Face IDs (showing {len(clusters)})")
    table.add_column("Face ID", style="cyan", justify="right")
    table.add_column(
        "Photos", style="magenta", justify="right"
    )
    table.add_column("Identity", style="green")
    table.add_column("Sample file", style="blue", no_wrap=True)

    for summary in clusters:
        table.add_row(*_format_cluster_row(summary))

    console.print(table)


@faces_app.command()
def name(
    cluster_label: int = typer.Argument(..., help="Cluster label to name."),
    identity_name: str = typer.Argument(..., help="Human name for the cluster."),
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
) -> None:
    """Assign a human name to a face cluster."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)
    service = FaceIdentityService(repository=repository)

    try:
        updated = service.name_cluster(cluster_label, identity_name)
    except ValueError as exc:
        console.print(f"[red]Invalid identity name:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        logger.exception("Failed to name cluster %d", cluster_label)
        console.print(f"[red]Failed to name cluster:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"Named cluster [cyan]{cluster_label}[/cyan] as "
        f"[green]{identity_name}[/green] "
        f"([magenta]{updated}[/magenta] face(s))."
    )


@faces_app.command()
def samples(
    face_id: int = typer.Argument(..., help="Face ID (cluster label) to sample."),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Number of random sample photos to show."
    ),
    include_heic: bool = typer.Option(
        False,
        "--include-heic",
        "-ih",
        help="Also include HEIC sample photos (JPEGs are always shown).",
    ),
    db_path: str | None = typer.Option(
        None, "--db", help="Path to the SQLite library database."
    ),
) -> None:
    """Show random sample JPEG photos for a face ID."""
    db = _get_db_path(db_path)
    repository = SqliteMediaRepository(db)
    service = FaceIdentityService(repository=repository)

    try:
        paths = service.get_sample_photos(
            face_id, limit=limit, include_heic=include_heic
        )
    except Exception as exc:
        logger.exception("Failed to load samples for face %d", face_id)
        console.print(f"[red]Failed to load samples:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not paths:
        console.print(
            f"[yellow]No JPEG photos found for face ID {face_id}.[/yellow]"
        )
        raise typer.Exit(0)

    table = Table(title=f"Sample JPEG photos for face ID {face_id}")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Path", style="blue", no_wrap=True)

    for i, path in enumerate(paths, start=1):
        table.add_row(str(i), path)

    console.print(table)
