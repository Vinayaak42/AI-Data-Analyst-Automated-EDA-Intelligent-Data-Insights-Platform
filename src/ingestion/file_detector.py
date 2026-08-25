from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".parquet": "parquet",
}


def detect_file_type(file_name: str) -> str:
    """
    Detect the type of uploaded data file.
    """

    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())

        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {supported}"
        )

    return SUPPORTED_EXTENSIONS[extension]