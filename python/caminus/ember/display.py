"""Make agent-controlled approval text visible without executing terminal controls."""


def terminal_text(value):
    return ''.join(char if char.isprintable() else ascii(char)[1:-1] for char in str(value))
