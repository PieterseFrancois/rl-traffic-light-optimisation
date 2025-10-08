from pathlib import Path


def load_int_list(filepath: str | Path) -> list[int]:
    """
    Load a text file containing one integer per line and return a list of ints.
    Blank lines are ignored. Raises ValueError on the first non-integer line.
    """
    path = Path(filepath)
    ints: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                ints.append(int(s))
            except ValueError as e:
                raise ValueError(f"Invalid integer on line {lineno}: {s!r}") from e
    return ints
