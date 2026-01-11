"""
Food Detection Pipeline Comparison
Research tool comparing LLM-only vs YOLO+LLM hybrid approaches.
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

    menu.add_row("1.", "Run System A — LLM-only pipeline")
    menu.add_row("2.", "Run System B — YOLO + LLM hybrid pipeline")
    menu.add_row("3.", "Exit")

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

    # Image info
    console.print(f"\n[bold]Image:[/bold] {result['meta']['image']}")
    console.print(f"[bold]Pipeline:[/bold] {result['meta']['pipeline']}")
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
                item.get("name", "Unknown"),
                f"[{state_color}]{state}[/{state_color}]"
            )

        console.print(table)
    else:
        console.print("[yellow]No food items detected.[/yellow]")

    # Raw JSON output (for research logging)
    console.print("\n[dim]─── Raw JSON Output ───[/dim]")
    console.print(f"[dim]{json.dumps(result, indent=2)}[/dim]")


def run_pipeline(pipeline_type: str):
    """Run selected pipeline with file picker."""
    # Select image
    image_path = select_image()

    if not image_path:
        console.print("[yellow]No image selected.[/yellow]")
        return

    path = Path(image_path)
    console.print(f"\n[bold]Selected:[/bold] {path.name}")

    # Run pipeline
    pipeline_name = "LLM-only (System A)" if pipeline_type == "llm" else "YOLO + LLM (System B)"

    with console.status(f"[cyan]Running {pipeline_name}...[/cyan]", spinner="dots"):
        try:
            if pipeline_type == "llm":
                from pipelines import llm_pipeline
                result = llm_pipeline.run(str(path))
            else:
                from pipelines import yolo_llm_pipeline
                result = yolo_llm_pipeline.run(str(path))

            display_results(result, pipeline_name)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main application loop."""
    while True:
        clear_screen()
        show_header()
        show_menu()

        choice = console.input("[bold green]Select option (1-3):[/bold green] ").strip()

        if choice == "1":
            console.print("\n[bold cyan]═══ System A: LLM-only Pipeline ═══[/bold cyan]")
            console.print("[dim]Sends full image to GPT-4o Vision for multi-item detection[/dim]\n")
            run_pipeline("llm")

        elif choice == "2":
            console.print("\n[bold cyan]═══ System B: YOLO + LLM Hybrid ═══[/bold cyan]")
            console.print("[dim]YOLO proposes regions → LLM identifies each crop → Results aggregated[/dim]\n")
            run_pipeline("yolo-llm")

        elif choice == "3":
            console.print("\n[cyan]Goodbye![/cyan]")
            break

        else:
            console.print("[red]Invalid option. Please select 1-3.[/red]")

        # Pause before returning to menu
        console.print()
        console.input("[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    main()
