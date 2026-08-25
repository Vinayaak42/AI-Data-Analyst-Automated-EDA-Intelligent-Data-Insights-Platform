from .file_detector import detect_file_type
from .csv_loader import load_csv
from .excel_loader import load_excel
from .json_loader import load_json
from .parquet_loader import load_parquet


def load_data(file):
    """
    Automatically detect and load a supported data file.
    """

    file_type = detect_file_type(file.name)

    if file_type == "csv":
        return load_csv(file)

    if file_type == "excel":
        return load_excel(file)

    if file_type == "json":
        return load_json(file)

    if file_type == "parquet":
        return load_parquet(file)

    raise ValueError(f"Unsupported file type: {file_type}")