import sys


def remove_lines_starting_with_xyz(file_path):
    """Removes lines starting with 'xyz' after trimming leading spaces."""
    try:
        # Read the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Rewrite the file, excluding lines that start with 'xyz' after stripping leading spaces
        with open(file_path, "w") as file:
            for line in lines:
                # Strip leading spaces and check if it starts with 'xyz'
                if not line.lstrip().startswith('log("SUCCESS: '):
                    file.write(line)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def main():
    # Check if there are file paths passed as arguments
    if len(sys.argv) < 2:
        print("Please provide at least one file as input.")
        sys.exit(1)

    # Iterate through the list of files provided in the command line arguments
    for file_path in sys.argv[1:]:
        remove_lines_starting_with_xyz(file_path)


if __name__ == "__main__":
    main()
