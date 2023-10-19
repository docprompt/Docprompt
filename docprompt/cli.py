"""Console script for docprompt."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("docprompt")
    click.echo("=" * len("docprompt"))
    click.echo("Documents and large language models")


if __name__ == "__main__":
    main()  # pragma: no cover
