import subprocess
import sys


def main():
    args = sys.argv[1:]
    module = "docprompt"  # Default module
    pytest_args = []

    # Parse arguments
    for arg in args:
        if arg.startswith("--mod="):
            module = arg.split("=")[1]
        else:
            pytest_args.append(arg)

    # Construct the pytest command
    command = [
        "pytest",
        f"--cov={module}",
        "--cov-report=term-missing",
    ] + pytest_args

    # Run the command
    subprocess.run(command)


if __name__ == "__main__":
    main()
