from src.data_preparation.data_preparation import (
    create_sequences,
    create_sequences_all,
    format_macro,
)


def main():
    # Format macroeconomic data
    macro = format_macro()
    # create_sequences_all(macro)


if __name__ == "__main__":
    main()
