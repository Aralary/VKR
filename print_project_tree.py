from pathlib import Path
import argparse


DEFAULT_IGNORES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "dist",
    "build",
}


def should_ignore(path: Path, ignore_names: set[str], show_hidden: bool) -> bool:
    name = path.name

    if name in ignore_names:
        return True

    if not show_hidden and name.startswith("."):
        return True

    return False


def build_tree(
    root: Path,
    prefix: str = "",
    ignore_names: set[str] | None = None,
    show_hidden: bool = False,
    max_depth: int | None = None,
    current_depth: int = 0,
) -> list[str]:
    if ignore_names is None:
        ignore_names = set()

    if max_depth is not None and current_depth >= max_depth:
        return []

    try:
        entries = sorted(
            root.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        return [prefix + "└── [Нет доступа]"]

    entries = [
        entry for entry in entries
        if not should_ignore(entry, ignore_names, show_hidden)
    ]

    lines = []

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(
                build_tree(
                    root=entry,
                    prefix=prefix + extension,
                    ignore_names=ignore_names,
                    show_hidden=show_hidden,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
            )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Печать дерева файлов проекта"
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Путь к проекту. По умолчанию текущая директория."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Показывать скрытые файлы и папки."
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Максимальная глубина обхода."
    )

    parser.add_argument(
        "--no-default-ignore",
        action="store_true",
        help="Не исключать стандартные папки вроде .git, venv, __pycache__."
    )

    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Дополнительные имена файлов или папок для исключения."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Файл для сохранения дерева. Например: tree.txt"
    )

    args = parser.parse_args()

    root = Path(args.path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Путь не найден: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Указанный путь не является папкой: {root}")

    ignore_names = set(args.ignore)

    if not args.no_default_ignore:
        ignore_names.update(DEFAULT_IGNORES)

    lines = [root.name]
    lines.extend(
        build_tree(
            root=root,
            ignore_names=ignore_names,
            show_hidden=args.all,
            max_depth=args.max_depth,
        )
    )

    result = "\n".join(lines)

    print(result)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result, encoding="utf-8")
        print(f"\nДерево сохранено в файл: {output_path.resolve()}")


if __name__ == "__main__":
    main()