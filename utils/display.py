"""
Terminal display utilities for SnapShelf.
Provides formatted output using the Rich library.
"""

from rich.console import Console
from rich.table import Table
from typing import List
from models.food_item import FoodItem

console = Console()


def display_items(items: List[FoodItem], show_row_numbers: bool = False):
    """
    Display food items in a formatted table.

    Args:
        items: List of FoodItem objects to display
        show_row_numbers: If True, show row numbers for selection instead of IDs
    """
    # Create table with columns
    table = Table(show_header=True, header_style="bold cyan")
    if show_row_numbers:
        table.add_column("#", style="bold yellow", width=4)
    else:
        table.add_column("ID", style="dim", width=8)
    table.add_column("Item", style="white")
    table.add_column("Category", style="yellow")
    table.add_column("Expiry", style="white")
    table.add_column("Status", style="white")

    # Add rows for each item
    for idx, item in enumerate(items, 1):
        # Format expiry date and days remaining
        if item.expiry_date:
            days = item.days_until_expiry()
            expiry_str = f"{item.expiry_date.strftime('%b %d')} ({days}d)"
        else:
            expiry_str = "No expiry"

        # Determine status with colored text based on expiry
        status = item.expiry_status()
        if status == "expired":
            status_str = "[bold red]Expired[/bold red]"
        elif status == "urgent":
            status_str = "[bold red]Urgent[/bold red]"
        elif status == "warning":
            status_str = "[bold orange1]Soon[/bold orange1]"
        elif status == "good":
            status_str = "[bold yellow]Good[/bold yellow]"
        else:
            status_str = "[bold green]Fresh[/bold green]"

        # Add row to table
        first_col = str(idx) if show_row_numbers else item.id
        table.add_row(
            first_col,
            item.name,
            item.category,
            expiry_str,
            status_str
        )

    # Print the table
    console.print(table)


def display_expiring(items: List[FoodItem]):
    """
    Display expiring items grouped by urgency.

    Args:
        items: List of FoodItem objects expiring soon
    """
    # Sort items by expiry date
    items_sorted = sorted(items, key=lambda x: x.expiry_date or date.max)

    # Group items by urgency
    expired = [i for i in items_sorted if i.days_until_expiry() < 0]
    next_3_days = [i for i in items_sorted if 0 <= i.days_until_expiry() <= 3]
    this_week = [i for i in items_sorted if 4 <= i.days_until_expiry() <= 7]

    # Display expired items
    console.print("[bold red]🔴 EXPIRED[/bold red]")
    if expired:
        for item in expired:
            days = abs(item.days_until_expiry())
            console.print(f"   {item.name} ({item.category}) - Expired {days} days ago")
    else:
        console.print("   None")

    console.print()

    # Display items expiring in next 3 days
    console.print("[bold orange1]🟠 NEXT 3 DAYS[/bold orange1]")
    if next_3_days:
        for idx, item in enumerate(next_3_days, 1):
            days = item.days_until_expiry()
            if days == 0:
                day_text = "today"
            elif days == 1:
                day_text = "tomorrow"
            else:
                day_text = f"in {days} days"
            console.print(f"   {idx}. {item.name} ({item.category}) - Expires {day_text}")
    else:
        console.print("   None")

    console.print()

    # Display items expiring this week
    console.print("[bold yellow]🟡 THIS WEEK[/bold yellow]")
    if this_week:
        for idx, item in enumerate(this_week, len(next_3_days) + 1):
            days = item.days_until_expiry()
            console.print(f"   {idx}. {item.name} ({item.category}) - Expires in {days} days")
    else:
        console.print("   None")


# Import date for use in display_expiring
from datetime import date
