"""
SnapShelf Console Application - Main CLI Entry Point
AI-powered food inventory management using YOLO and Vision LLM.
"""

import sys
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from datetime import date
from typing import List
from tkinter import Tk, filedialog

from services.scan_service import ScanService
from storage.inventory import InventoryStorage
from utils.display import display_items, display_expiring

# Initialize Typer app and Rich console
app = typer.Typer(help="SnapShelf - AI Food Inventory Manager")
console = Console()
storage = InventoryStorage()


@app.command()
def scan(image: Path = typer.Argument(..., help="Path to food image")):
    """
    Scan an image and add detected food items to inventory.

    Args:
        image: Path to the image file containing food items
    """
    # Validate image file exists
    if not image.exists():
        console.print(f"[red]Error: File not found: {image}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Scanning:[/bold] {image.name}\n")

    # Execute scan with loading indicator
    with console.status("Detecting objects..."):
        scan_service = ScanService()
        items = scan_service.scan_image(str(image))

    # Display results
    if items:
        console.print(f"[green]✓ Added {len(items)} items to inventory:[/green]\n")
        display_items(items)
    else:
        console.print("[yellow]No food items detected in image.[/yellow]")


@app.command()
def list(
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    sort: str = typer.Option("expiry", "--sort", "-s", help="Sort by: expiry, name, added"),
    status: str = typer.Option("active", "--status", help="Filter: active, consumed, discarded")
):
    """
    List all items in inventory.

    Args:
        category: Filter items by category (optional)
        sort: Sort order (expiry, name, or added)
        status: Filter by status (active, consumed, discarded)
    """
    # Get all items with status filter
    items = storage.get_all(status=status)

    # Apply category filter if specified
    if category:
        items = [i for i in items if i.category.lower() == category.lower()]

    # Sort items based on specified field
    if sort == "expiry":
        items.sort(key=lambda x: x.expiry_date or date.max)
    elif sort == "name":
        items.sort(key=lambda x: x.name.lower())
    elif sort == "added":
        items.sort(key=lambda x: x.added_at, reverse=True)

    # Display results
    if items:
        console.print(f"\n[bold]Inventory ({len(items)} items)[/bold]\n")
        display_items(items)
    else:
        console.print("\n[yellow]Inventory is empty.[/yellow]")


@app.command()
def expiring(days: int = typer.Option(7, "--days", "-d", help="Days to look ahead")):
    """
    Show items expiring soon.

    Args:
        days: Number of days to look ahead for expiring items
    """
    # Get expiring items
    items = storage.get_expiring(days=days)

    # Display results
    if items:
        console.print(f"\n[bold]⚠️  Items Expiring (next {days} days)[/bold]\n")
        display_expiring(items)
    else:
        console.print(f"\n[green]No items expiring in the next {days} days.[/green]")


@app.command()
def consume(ids: List[str] = typer.Argument(..., help="Item ID(s) to mark as consumed")):
    """
    Mark item(s) as consumed.

    Args:
        ids: List of item IDs to mark as consumed
    """
    # Process each item ID
    for item_id in ids:
        item = storage.get_by_id(item_id)
        if item:
            storage.update(item_id, {"status": "consumed"})
            console.print(f"[green]✓ Consumed:[/green] {item.name}")
        else:
            console.print(f"[red]Item not found: {item_id}[/red]")


@app.command()
def discard(ids: List[str] = typer.Argument(..., help="Item ID(s) to mark as discarded")):
    """
    Mark item(s) as discarded (wasted).

    Args:
        ids: List of item IDs to mark as discarded
    """
    # Process each item ID
    for item_id in ids:
        item = storage.get_by_id(item_id)
        if item:
            storage.update(item_id, {"status": "discarded"})
            console.print(f"[yellow]✓ Discarded:[/yellow] {item.name}")
        else:
            console.print(f"[red]Item not found: {item_id}[/red]")


@app.command()
def remove(ids: List[str] = typer.Argument(..., help="Item ID(s) to remove")):
    """
    Remove item(s) from inventory.

    Args:
        ids: List of item IDs to permanently remove
    """
    # Process each item ID
    for item_id in ids:
        item = storage.get_by_id(item_id)
        if item:
            storage.remove(item_id)
            console.print(f"[green]✓ Removed:[/green] {item.name}")
        else:
            console.print(f"[red]Item not found: {item_id}[/red]")


@app.command()
def clear(confirm: bool = typer.Option(False, "--confirm", "-y", help="Skip confirmation")):
    """
    Clear all items from inventory.

    Args:
        confirm: Skip confirmation prompt if True
    """
    # Get count of items to clear
    if not confirm:
        items = storage.get_all(status=None)
        if not typer.confirm(f"Clear all {len(items)} items?"):
            raise typer.Abort()

    # Clear inventory
    storage.clear()
    console.print("[green]✓ Inventory cleared[/green]")


def interactive_menu():
    """
    Run SnapShelf in interactive menu mode.
    Displays a menu and prompts user for actions.
    """
    while True:
        # Clear screen (optional - comment out if you prefer)
        console.clear()

        # Display header
        header = Panel(
            "[bold cyan]SnapShelf - AI Food Inventory Manager[/bold cyan]\n"
            "[dim]Smart food tracking with AI-powered detection[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(header)
        console.print()

        # Display menu options
        menu = Table(show_header=False, box=None, padding=(0, 2))
        menu.add_column("Option", style="bold yellow", width=4)
        menu.add_column("Action", style="white")

        menu.add_row("1.", "Scan food image")
        menu.add_row("2.", "View inventory")
        menu.add_row("3.", "View expiring items")
        menu.add_row("4.", "Remove item")
        menu.add_row("5.", "Clear all inventory")
        menu.add_row("6.", "Exit")

        console.print(menu)
        console.print()

        # Get user choice
        choice = console.input("[bold green]Select option (1-6):[/bold green] ").strip()
        console.print()

        # Process user choice
        if choice == "1":
            # Scan image - Open file picker dialog
            console.print("[cyan]Opening file picker...[/cyan]")

            # Initialize tkinter and hide root window
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            # Open file dialog
            image_path = filedialog.askopenfilename(
                title="Select Food Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )

            # Clean up tkinter
            root.destroy()

            if image_path:
                path = Path(image_path)
                console.print(f"\n[bold]Scanning:[/bold] {path.name}\n")
                with console.status("Detecting objects..."):
                    scan_service = ScanService()
                    items = scan_service.scan_image_preview(str(path))

                if items:
                    # Review loop - let user edit/delete items before saving
                    while True:
                        console.print(f"[green]✓ Detected {len(items)} item(s):[/green]\n")
                        display_items(items, show_row_numbers=True)
                        console.print()
                        console.print("[dim]  Press Enter to save all[/dim]")
                        console.print("[dim]  Enter number to edit item (e.g. '1')[/dim]")
                        console.print("[dim]  Enter 'd' + number to delete (e.g. 'd1')[/dim]")
                        console.print()

                        action = console.input("[cyan]>[/cyan] ").strip().lower()

                        if action == "":
                            # Save all items
                            if items:
                                scan_service.save_items(items)
                                console.print(f"\n[green]✓ Added {len(items)} item(s) to inventory![/green]")
                            else:
                                console.print("[yellow]No items to save.[/yellow]")
                            break

                        elif action.startswith("d") and action[1:].isdigit():
                            # Delete item
                            idx = int(action[1:])
                            if 1 <= idx <= len(items):
                                removed = items.pop(idx - 1)
                                console.print(f"[yellow]✓ Removed:[/yellow] {removed.name}\n")
                            else:
                                console.print(f"[red]Invalid number. Enter 1-{len(items)}[/red]\n")

                        elif action.isdigit():
                            # Edit item
                            idx = int(action)
                            if 1 <= idx <= len(items):
                                item = items[idx - 1]
                                console.print(f"\n[bold]Editing: {item.name}[/bold]")
                                console.print("[dim]Press Enter to keep current value[/dim]\n")

                                # Edit name
                                new_name = console.input(f"  Name [{item.name}]: ").strip()
                                if new_name:
                                    item.name = new_name

                                # Edit quantity
                                unit_hint = f" ({item.unit})" if item.unit != "unit" else ""
                                qty_display = int(item.quantity) if item.quantity == int(item.quantity) else item.quantity
                                new_qty = console.input(f"  Quantity{unit_hint} [{qty_display}]: ").strip()
                                if new_qty.replace('.', '', 1).isdigit() and float(new_qty) > 0:
                                    item.quantity = float(new_qty)

                                # Edit expiry date
                                current_expiry = item.expiry_date.strftime("%Y-%m-%d") if item.expiry_date else "none"
                                new_expiry = console.input(f"  Expiry date [{current_expiry}]: ").strip()
                                if new_expiry:
                                    try:
                                        from datetime import datetime
                                        item.expiry_date = datetime.strptime(new_expiry, "%Y-%m-%d").date()
                                    except ValueError:
                                        console.print("[red]  Invalid date format. Use YYYY-MM-DD[/red]")

                                console.print(f"[green]✓ Updated {item.name}[/green]\n")
                            else:
                                console.print(f"[red]Invalid number. Enter 1-{len(items)}[/red]\n")

                        else:
                            console.print("[red]Invalid input.[/red]\n")
                else:
                    console.print("[yellow]No food items detected in image.[/yellow]")
            else:
                console.print("[yellow]No file selected.[/yellow]")

        elif choice == "2":
            # View inventory (auto-consolidate duplicates first)
            merged = storage.consolidate()
            if merged > 0:
                console.print(f"[dim]Consolidated {merged} duplicate item(s)[/dim]\n")
            items = storage.get_all(status="active")
            if items:
                items.sort(key=lambda x: x.expiry_date or date.max)
                console.print(f"[bold]Inventory ({len(items)} items)[/bold]\n")
                display_items(items)
            else:
                console.print("[yellow]Inventory is empty.[/yellow]")

        elif choice == "3":
            # View expiring items
            days = console.input("[cyan]Days to look ahead (default 7):[/cyan] ").strip()
            days = int(days) if days.isdigit() else 7

            items = storage.get_expiring(days=days)
            if items:
                console.print(f"[bold]⚠️  Items Expiring (next {days} days)[/bold]\n")
                display_expiring(items)
            else:
                console.print(f"[green]No items expiring in the next {days} days.[/green]")

        elif choice == "4":
            # Remove item (with reason)
            items = storage.get_all(status="active")
            if not items:
                console.print("[yellow]No items in inventory.[/yellow]")
            else:
                items.sort(key=lambda x: x.expiry_date or date.max)
                console.print("[bold]Select item to remove:[/bold]\n")
                display_items(items, show_row_numbers=True)
                console.print()
                selection = console.input("[cyan]Enter item number (or 'c' to cancel):[/cyan] ").strip()
                if selection.lower() != 'c' and selection.isdigit():
                    idx = int(selection)
                    if 1 <= idx <= len(items):
                        item = items[idx - 1]
                        # Ask for action
                        unit_str = f" {item.unit}" if item.unit != "unit" else ""
                        qty_display = int(item.quantity) if item.quantity == int(item.quantity) else item.quantity
                        console.print(f"\n[bold]What happened to {item.name}? (Current: {qty_display}{unit_str})[/bold]\n")
                        console.print("  1. Use some (reduce quantity)")
                        console.print("  2. Consumed all")
                        console.print("  3. Discarded (thrown away)")
                        console.print("  4. Delete (remove from history)")
                        console.print("  c. Cancel")
                        console.print()
                        action = console.input("[cyan]Select option:[/cyan] ").strip()
                        if action == "1":
                            # Use some - reduce quantity
                            amount_input = console.input(f"[cyan]How much did you use?{unit_str}: [/cyan]").strip()
                            if amount_input.replace('.', '', 1).isdigit():
                                amount = float(amount_input)
                                if amount >= item.quantity:
                                    # Used all or more - mark as consumed
                                    storage.update(item.id, {"status": "consumed"})
                                    console.print(f"[green]✓ All consumed:[/green] {item.name}")
                                elif amount > 0:
                                    # Reduce quantity
                                    new_qty = item.quantity - amount
                                    storage.update(item.id, {"quantity": new_qty})
                                    new_display = int(new_qty) if new_qty == int(new_qty) else new_qty
                                    console.print(f"[green]✓ Updated:[/green] {item.name} ({new_display}{unit_str} remaining)")
                                else:
                                    console.print("[red]Amount must be greater than 0.[/red]")
                            else:
                                console.print("[red]Invalid amount.[/red]")
                        elif action == "2":
                            storage.update(item.id, {"status": "consumed"})
                            console.print(f"[green]✓ Marked as consumed:[/green] {item.name}")
                        elif action == "3":
                            storage.update(item.id, {"status": "discarded"})
                            console.print(f"[yellow]✓ Marked as discarded:[/yellow] {item.name}")
                        elif action == "4":
                            storage.remove(item.id)
                            console.print(f"[green]✓ Deleted:[/green] {item.name}")
                        elif action.lower() != 'c':
                            console.print("[red]Invalid option.[/red]")
                    else:
                        console.print(f"[red]Invalid selection. Enter 1-{len(items)}[/red]")
                elif selection.lower() != 'c':
                    console.print("[red]Invalid input.[/red]")

        elif choice == "5":
            # Clear inventory
            items = storage.get_all(status=None)
            if items:
                confirm = console.input(f"[yellow]Clear all {len(items)} items? (y/N):[/yellow] ").strip().lower()
                if confirm == 'y':
                    storage.clear()
                    console.print("[green]✓ Inventory cleared[/green]")
                else:
                    console.print("[dim]Cancelled[/dim]")
            else:
                console.print("[yellow]Inventory is already empty.[/yellow]")

        elif choice == "6":
            # Exit
            console.print("[cyan]Thank you for using SnapShelf![/cyan]")
            break

        else:
            console.print("[red]Invalid option. Please select 1-6.[/red]")

        # Pause before returning to menu
        console.print()
        console.input("[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    # Check if running with command-line arguments
    if len(sys.argv) > 1:
        # Run as traditional CLI with arguments
        app()
    else:
        # Run in interactive menu mode
        interactive_menu()
