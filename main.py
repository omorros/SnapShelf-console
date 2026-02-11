"""
Experiment 2: End-to-End Pipeline Comparison (14-class Fruit/Veg Inventory)

Compares three fundamentally different pipelines for the same 14-class inventory task:
- Pipeline A (vlm):      Image -> GPT-4o-mini (constrained to 14 labels) -> inventory
- Pipeline B (yolo-14):  Image -> 14-class YOLO -> boxes + labels -> inventory
- Pipeline C (yolo-cnn): Image -> 1-class objectness YOLO -> crops -> CNN -> inventory

Usage:
    python main.py                                   Interactive menu
    python main.py vlm <image_path>                  Pipeline A
    python main.py yolo-14 <image_path>              Pipeline B
    python main.py yolo-cnn <image_path>             Pipeline C
    python main.py evaluate --images <dir> --labels <dir>   Evaluate all pipelines
    python main.py train yolo-14                     Train 14-class YOLO
    python main.py train yolo-obj                    Train objectness YOLO
    python main.py --validate                        Validate environment
"""

import sys
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_pipeline(pipeline_type: str, image_path: str) -> dict:
    """
    Run selected pipeline and return result.

    Args:
        pipeline_type: "vlm", "yolo-14", or "yolo-cnn"
        image_path: Path to image file

    Returns:
        PipelineResult dict
    """
    if pipeline_type == "vlm":
        from pipelines.vlm_pipeline import run
        return run(image_path)
    elif pipeline_type == "yolo-14":
        from pipelines.yolo_pipeline import run
        return run(image_path)
    elif pipeline_type == "yolo-cnn":
        from pipelines.yolo_cnn_pipeline import run
        return run(image_path)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_type}")


# =============================================================================
# CLI MODE
# =============================================================================

def cli_run(pipeline: str, image_path: str):
    """Run a pipeline on a single image and print JSON."""
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    from config import init_experiment
    init_experiment(require_openai=(pipeline == "vlm"))

    result = run_pipeline(pipeline, image_path)
    print(json.dumps(result, indent=2))


def cli_evaluate(args: list):
    """Run evaluation across all pipelines."""
    import argparse

    parser = argparse.ArgumentParser(prog="main.py evaluate")
    parser.add_argument("--images", required=True, help="Directory with test images")
    parser.add_argument("--labels", required=True, help="Directory with YOLO .txt labels")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        help="Pipelines to evaluate (vlm, yolo-14, yolo-cnn)")
    parser.add_argument("--output", default=None, help="Output directory")
    parsed = parser.parse_args(args)

    from config import init_experiment
    init_experiment()

    from evaluation.evaluate_runner import run_evaluation
    from evaluation.report import generate_report

    all_results = run_evaluation(
        images_dir=parsed.images,
        labels_dir=parsed.labels,
        pipelines=parsed.pipelines,
        output_dir=parsed.output,
    )

    if all_results:
        generate_report(all_results, output_dir=parsed.output)


def cli_train(model_type: str, extra_args: list):
    """Run training for a YOLO model."""
    if model_type == "yolo-14":
        from training.train_yolo_14class import train
        import argparse
        parser = argparse.ArgumentParser(prog="main.py train yolo-14")
        parser.add_argument("--data", default="data/yolo_14class.yaml")
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch", type=int, default=None)
        parsed = parser.parse_args(extra_args)
        train(data_yaml=parsed.data, epochs=parsed.epochs, batch=parsed.batch)

    elif model_type == "yolo-obj":
        from training.train_yolo_objectness import train
        import argparse
        parser = argparse.ArgumentParser(prog="main.py train yolo-obj")
        parser.add_argument("--data", default="data/yolo_objectness.yaml")
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch", type=int, default=None)
        parsed = parser.parse_args(extra_args)
        train(data_yaml=parsed.data, epochs=parsed.epochs, batch=parsed.batch)

    else:
        print(f"Unknown model type: {model_type}", file=sys.stderr)
        print("Use: yolo-14 | yolo-obj", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def show_header():
    """Display application header."""
    header = Panel(
        "[bold cyan]Experiment 2: End-to-End Pipeline Comparison[/bold cyan]\n"
        "[dim]14-class Fruit/Veg Inventory — VLM vs YOLO vs YOLO+CNN[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(header)
    console.print()


def show_menu():
    """Display main menu options."""
    menu = Table(show_header=False, box=None, padding=(0, 2))
    menu.add_column("Option", style="bold yellow", width=4)
    menu.add_column("Description", style="white")

    menu.add_row("1.", "Pipeline A — VLM-only (GPT-4o-mini, 14 labels)")
    menu.add_row("2.", "Pipeline B — YOLO end-to-end (14-class YOLO)")
    menu.add_row("3.", "Pipeline C — YOLO + CNN (objectness YOLO + CNN)")
    menu.add_row("4.", "Evaluate all pipelines on test set")
    menu.add_row("5.", "Train YOLO model")
    menu.add_row("6.", "Validate environment")
    menu.add_row("7.", "Exit")

    console.print(menu)
    console.print()


def select_image() -> str | None:
    """Open file picker dialog to select an image."""
    console.print("[cyan]Opening file picker...[/cyan]")
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return image_path if image_path else None


def display_results(result: dict, pipeline_name: str):
    """Display pipeline results in a formatted view."""
    console.print()

    console.print(Panel(
        f"[bold green]Results — {pipeline_name}[/bold green]",
        box=box.ROUNDED,
        border_style="green"
    ))

    meta = result.get("meta", {})
    console.print(f"\n[bold]Image:[/bold] {meta.get('image', 'N/A')}")
    console.print(f"[bold]Pipeline:[/bold] {meta.get('pipeline', 'N/A')}")
    console.print(f"[bold]Runtime:[/bold] {meta.get('runtime_ms', 0):.0f} ms")

    timing = meta.get("timing_breakdown", {})
    if timing:
        console.print("\n[bold]Timing Breakdown:[/bold]")
        for key, value in timing.items():
            if isinstance(value, float):
                console.print(f"  - {key}: {value:.2f} ms")
            else:
                console.print(f"  - {key}: {value}")

    if meta.get("detections_count") is not None:
        console.print(f"[bold]Detections:[/bold] {meta['detections_count']}")

    console.print()

    inventory = result.get("inventory", {})
    if inventory:
        table = Table(
            title=f"[bold]Inventory ({sum(inventory.values())} items)[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Class", style="white", min_width=20)
        table.add_column("Count", style="yellow", justify="center")

        for cls_name, count in sorted(inventory.items()):
            table.add_row(cls_name, str(count))

        console.print(table)
    else:
        console.print("[yellow]No items detected.[/yellow]")

    console.print(f"\n[dim]{json.dumps(result, indent=2)}[/dim]")


def interactive_run_pipeline(pipeline_type: str):
    """Run pipeline interactively with file picker."""
    image_path = select_image()
    if not image_path:
        console.print("[yellow]No image selected.[/yellow]")
        return

    console.print(f"\n[bold]Selected:[/bold] {Path(image_path).name}")

    names = {
        "vlm": "VLM-only (Pipeline A)",
        "yolo-14": "YOLO end-to-end (Pipeline B)",
        "yolo-cnn": "YOLO + CNN (Pipeline C)",
    }
    name = names.get(pipeline_type, pipeline_type)

    with console.status(f"[cyan]Running {name}...[/cyan]", spinner="dots"):
        try:
            result = run_pipeline(pipeline_type, image_path)
            display_results(result, name)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def interactive_evaluate():
    """Run evaluation interactively."""
    images_dir = console.input("[bold]Test images directory: [/bold]").strip()
    labels_dir = console.input("[bold]Labels directory: [/bold]").strip()

    if not Path(images_dir).exists():
        console.print(f"[red]Not found: {images_dir}[/red]")
        return
    if not Path(labels_dir).exists():
        console.print(f"[red]Not found: {labels_dir}[/red]")
        return

    from config import init_experiment
    init_experiment()

    from evaluation.evaluate_runner import run_evaluation
    from evaluation.report import generate_report

    all_results = run_evaluation(images_dir=images_dir, labels_dir=labels_dir)
    if all_results:
        generate_report(all_results)


def interactive_train():
    """Select and run training interactively."""
    console.print("[bold]Select model to train:[/bold]")
    console.print("  1. 14-class YOLO (Pipeline B)")
    console.print("  2. Objectness YOLO (Pipeline C)")
    choice = console.input("[bold green]Select (1-2): [/bold green]").strip()

    if choice == "1":
        cli_train("yolo-14", [])
    elif choice == "2":
        cli_train("yolo-obj", [])
    else:
        console.print("[red]Invalid choice.[/red]")


def validate_environment():
    """Validate environment and display configuration."""
    from config import validate_environment as _validate, CONFIG, CLASSES

    console.print("[cyan]Validating environment...[/cyan]\n")
    result = _validate()

    table = Table(title="Experiment Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("VLM Model", CONFIG.vlm_model)
    table.add_row("VLM Temperature", str(CONFIG.vlm_temperature))
    table.add_row("YOLO Confidence", str(CONFIG.yolo_conf_threshold))
    table.add_row("YOLO Max Detections", str(CONFIG.yolo_max_detections))
    table.add_row("CNN Model", CONFIG.cnn_model_name)
    table.add_row("Classes", str(len(CLASSES)))
    table.add_row("Random Seed", str(CONFIG.random_seed))

    console.print(table)
    console.print()

    if result["valid"]:
        console.print("[green][OK] Environment valid[/green]")
    else:
        console.print("[red][FAIL] Environment invalid[/red]")
        for error in result["errors"]:
            console.print(f"  [red]- {error}[/red]")

    for warning in result.get("warnings", []):
        console.print(f"  [yellow][WARN] {warning}[/yellow]")

    return result["valid"]


def interactive_mode():
    """Main interactive application loop."""
    from config import init_experiment
    try:
        init_experiment()
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    while True:
        console.clear()
        show_header()
        show_menu()

        choice = console.input("[bold green]Select option (1-7):[/bold green] ").strip()

        if choice == "1":
            console.print("\n[bold cyan]Pipeline A: VLM-only[/bold cyan]")
            interactive_run_pipeline("vlm")
        elif choice == "2":
            console.print("\n[bold cyan]Pipeline B: YOLO end-to-end[/bold cyan]")
            interactive_run_pipeline("yolo-14")
        elif choice == "3":
            console.print("\n[bold cyan]Pipeline C: YOLO + CNN[/bold cyan]")
            interactive_run_pipeline("yolo-cnn")
        elif choice == "4":
            console.print("\n[bold cyan]Evaluate All Pipelines[/bold cyan]")
            interactive_evaluate()
        elif choice == "5":
            console.print("\n[bold cyan]Train Model[/bold cyan]")
            interactive_train()
        elif choice == "6":
            console.print("\n[bold cyan]Environment Validation[/bold cyan]")
            validate_environment()
        elif choice == "7":
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        else:
            console.print("[red]Invalid option. Please select 1-7.[/red]")

        console.print()
        console.input("[dim]Press Enter to continue...[/dim]")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def print_usage():
    """Print usage and exit."""
    print(__doc__, file=sys.stderr)
    sys.exit(1)


def main():
    """Main entrypoint."""
    if len(sys.argv) == 1:
        interactive_mode()
        return

    cmd = sys.argv[1].lower()

    if cmd in ("--help", "-h"):
        print_usage()

    if cmd == "--validate":
        valid = validate_environment()
        sys.exit(0 if valid else 1)

    # Pipeline commands
    if cmd in ("vlm", "yolo-14", "yolo-cnn"):
        if len(sys.argv) < 3:
            print(f"Error: {cmd} requires an image path", file=sys.stderr)
            print_usage()
        cli_run(cmd, sys.argv[2])
        return

    # Evaluate command
    if cmd == "evaluate":
        cli_evaluate(sys.argv[2:])
        return

    # Train command
    if cmd == "train":
        if len(sys.argv) < 3:
            print("Error: train requires a model type (yolo-14 | yolo-obj)", file=sys.stderr)
            sys.exit(1)
        cli_train(sys.argv[2], sys.argv[3:])
        return

    print(f"Error: Unknown command '{cmd}'", file=sys.stderr)
    print_usage()


if __name__ == "__main__":
    main()
