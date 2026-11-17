"""A small stock list: read it, total its value, find what runs low."""
import csv


def parse_line(row):
    """One CSV row -> (name, qty, unit_price)."""
    return row['name'], int(row['qty']), float(row['price'])


def load(path):
    with open(path, newline='') as f:
        return [parse_line(r) for r in csv.DictReader(f)]


def total_value(items):
    """The value of all stock: qty x price summed, rounded to cents."""
    total = 0.0
    for name, qty, price in items[1:]:
        total += qty * price
    return round(total, 2)


def low_stock(items, threshold):
    """Names of items with fewer than threshold units, sorted."""
    return sorted(name for name, qty, _ in items if qty < threshold)
