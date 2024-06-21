"""Generate the code reference pages and navigation."""

from pathlib import Path
import mkdocs_gen_files
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nav = mkdocs_gen_files.Nav()

root_path = Path("docprompt")
logger.debug(f"Searching for Python files in: {root_path.absolute()}")

python_files = list(root_path.rglob("*.py"))
logger.debug(f"Found {len(python_files)} Python files")

if not python_files:
    logger.warning("No Python files found. Ensure the 'docprompt' directory exists and contains .py files.")
    nav["No modules found"] = "no_modules.md"
    with mkdocs_gen_files.open("reference/no_modules.md", "w") as fd:
        fd.write("# No Modules Found\n\nNo Python modules were found in the 'docprompt' directory.")
else:
    for path in sorted(python_files):
        module_path = path.relative_to(root_path).with_suffix("")
        doc_path = path.relative_to(root_path).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        # Handle empty parts
        if not parts:
            logger.warning(f"Empty parts for file: {path}. Skipping this file.")
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: docprompt.{ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    logger.debug("Navigation structure:")
    logger.debug(nav.build_literate_nav())

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

