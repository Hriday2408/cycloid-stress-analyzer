"""Console report helper with the KEY / diagnostic marking convention."""

W = 72
_LEGEND = (
    "  [KEY] -> read: pass/fail results and numbers that matter\n"
    "  [.. ] -> intermediate diagnostic, safe to skip"
)


def header(title: str):
    print("=" * W)
    print(title)
    print("=" * W)
    print(_LEGEND)
    print("-" * W)


def key(label: str, value, note: str = ""):
    print(f"  [KEY] {label:<36} {value} {note}")


def info(label: str, value, note: str = ""):
    print(f"  [.. ] {label:<36} {value} {note}")


def rule():
    print("-" * W)


def ok_fail(flag: bool) -> str:
    return "PASS" if flag else "FAIL"