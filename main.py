"""
Food Detection Pipeline Comparison
Research tool comparing three vision pre-processing approaches for food recognition.

Usage:
    python main.py                           Interactive menu
    python main.py llm <image_path>          Pipeline A: LLM-only (baseline)
    python main.py yolo <image_path>         Pipeline B: Class-agnostic YOLO + LLM
    python main.py yolo-world <image_path>   Pipeline C: YOLO-World + LLM
"""

import sys
import json
from pathlib import Path
from tkinter import Tk, filedialog

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# =============================================================================
# SHARED PIPELINE RUNNER (used by both CLI and Interactive modes)
# =============================================================================

def run_pipeline(pipeline_type: str, image_path: str) -> dict:
    """
    Run selected pipeline and return result.
    Single source of truth for both CLI and interactive modes.

    Args:
        pipeline_type: "llm", "yolo", or "yolo-world"
        image_path: Path to image file

    Returns:
        Pipeline result dict
    """
    if pipeline_type == "llm":
        from pipelines import llm_pipeline
        return llm_pipeline.run(image_path)
    elif pipeline_type == "yolo":
        from pipelines import yolo_agnostic_pipeline
        return yolo_agnostic_pipeline.run(image_path)
    else:  # yolo-world
        from pipelines import yolo_world_pipeline
        return yolo_world_pipeline.run(image_path)


# =============================================================================
# CLI MODE
# =============================================================================

def cli_mode(pipeline: str, image_path: str):
    """Run in CLI mode - output JSON to stdout."""
    # Validate image exists
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    result = run_pipeline(pipeline, image_path)

    # Output JSON
    print(json.dumps(result, indent=2))


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    console.clear()


def show_header():
    """Display application header."""
    header = Panel(
        "[bold cyan]Food Detection Pipeline Comparison[/bold cyan]\n"
        "[dim]Research tool for comparing detection approaches[/dim]",
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

    menu.add_row("1.", "Pipeline A — LLM-only (baseline)")
    menu.add_row("2.", "Pipeline B — Class-agnostic YOLO + LLM")
    menu.add_row("3.", "Pipeline C — YOLO-World + LLM")
    menu.add_row("4.", "Exit")

    console.print(menu)
    console.print()


def select_image() -> str | None:
    """Open file picker dialog to select an image."""
    console.print("[cyan]Opening file picker...[/cyan]")

    # Initialize tkinter (hidden window)
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open file dialog
    image_path = filedialog.askopenfilename(
        title="Select Food Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return image_path if image_path else None


def display_results(result: dict, pipeline_name: str):
    """Display pipeline results in a nice format."""
    console.print()

    # Header
    console.print(Panel(
        f"[bold green]Results — {pipeline_name}[/bold green]",
        box=box.ROUNDED,
        border_style="green"
    ))

    # Meta info
    meta = result.get("meta", {})
    console.print(f"\n[bold]Image:[/bold] {meta.get('image', 'N/A')}")
    console.print(f"[bold]Pipeline:[/bold] {meta.get('pipeline', 'N/A')}")
    console.print(f"[bold]Runtime:[/bold] {meta.get('runtime_ms', 0):.0f} ms")

    if meta.get("pipeline") in ("yolo", "yolo-world"):
        fallback = meta.get("fallback_used", False)
        fallback_str = "[yellow]Yes[/yellow]" if fallback else "[green]No[/green]"
        console.print(f"[bold]Fallback used:[/bold] {fallback_str}")
        detections = meta.get("detections_count", "N/A")
        console.print(f"[bold]Detections:[/bold] {detections}")

    console.print()

    # Items table
    items = result.get("items", [])

    if items:
        table = Table(
            title=f"[bold]Detected Items ({len(items)})[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("#", style="dim", width=3, justify="center")
        table.add_column("Item Name", style="white", min_width=20)
        table.add_column("State", style="yellow", justify="center")

        for i, item in enumerate(items, 1):
            state = item.get("state", "unknown")
            state_color = {
                "fresh": "green",
                "packaged": "blue",
                "cooked": "yellow",
                "unknown": "dim"
            }.get(state, "white")

            table.add_row(
                str(i),
                item.get("name", "unknown"),
                f"[{state_color}]{state}[/{state_color}]"
            )

        console.print(table)
    else:
        console.print("[yellow]No food items detected.[/yellow]")

    # Raw JSON output (for research logging)
    console.print("\n[dim]─── Raw JSON Output ───[/dim]")
    console.print(f"[dim]{json.dumps(result, indent=2)}[/dim]")


def interactive_run_pipeline(pipeline_type: str):
    """Run pipeline in interactive mode with file picker and display."""
    # Select image
    image_path = select_image()

    if not image_path:
        console.print("[yellow]No image selected.[/yellow]")
        return

    path = Path(image_path)
    console.print(f"\n[bold]Selected:[/bold] {path.name}")

    # Run pipeline
    pipeline_names = {
        "llm": "LLM-only (Pipeline A)",
        "yolo": "Class-agnostic YOLO + LLM (Pipeline B)",
        "yolo-world": "YOLO-World + LLM (Pipeline C)"
    }
    pipeline_name = pipeline_names.get(pipeline_type, pipeline_type)

    with console.status(f"[cyan]Running {pipeline_name}...[/cyan]", spinner="dots"):
        try:
            result = run_pipeline(pipeline_type, str(path))
            display_results(result, pipeline_name)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def interactive_mode():
    """Main interactive application loop."""
    while True:
        clear_screen()
        show_header()
        show_menu()

        choice = console.input("[bold green]Select option (1-4):[/bold green] ").strip()

        if choice == "1":
            console.print("\n[bold cyan]═══ Pipeline A: LLM-only ═══[/bold cyan]")
            console.print("[dim]Sends full image to GPT-4o Vision for multi-item detection[/dim]\n")
            interactive_run_pipeline("llm")

        elif choice == "2":
            console.print("\n[bold cyan]═══ Pipeline B: Class-agnostic YOLO + LLM ═══[/bold cyan]")
            console.print("[dim]YOLOv8 proposes regions (labels ignored) → LLM identifies each crop[/dim]\n")
            interactive_run_pipeline("yolo")

        elif choice == "3":
            console.print("\n[bold cyan]═══ Pipeline C: YOLO-World + LLM ═══[/bold cyan]")
            console.print("[dim]YOLO-World proposes regions (semantic prompts) → LLM identifies each crop[/dim]\n")
            interactive_run_pipeline("yolo-world")

        elif choice == "4":
            console.print("\n[cyan]Goodbye![/cyan]")
            break

        else:
            console.print("[red]Invalid option. Please select 1-4.[/red]")

        # Pause before returning to menu
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
    """Main entrypoint - route to CLI or Interactive mode."""
    if len(sys.argv) == 1:
        # No args: interactive mode
        interactive_mode()

    elif len(sys.argv) >= 2:
        pipeline = sys.argv[1].lower()

        # Validate pipeline
        if pipeline not in ("llm", "yolo", "yolo-world"):
            print(f"Error: Unknown pipeline '{pipeline}'", file=sys.stderr)
            print("Use 'llm', 'yolo', or 'yolo-world'", file=sys.stderr)
            sys.exit(1)

        # CLI mode requires image path
        if len(sys.argv) < 3:
            print(f"Error: {pipeline} requires an image path", file=sys.stderr)
            print_usage()

        cli_mode(pipeline, sys.argv[2])

    else:
        print_usage()


if __name__ == "__main__":
    main()
