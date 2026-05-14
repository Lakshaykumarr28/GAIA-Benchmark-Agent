import os

from smolagents import Tool


class ExcelReadTool(Tool):
    """
    Reads an Excel (.xlsx, .xls) or CSV file and returns its contents as a
    formatted string. Useful for GAIA tasks involving spreadsheet data.
    """

    name = "read_excel_file"
    description = (
        "Reads an Excel (.xlsx/.xls) or CSV file and returns all sheet data "
        "as a formatted text table. Use this when download_task_file returns "
        "a path to a spreadsheet file."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Local path to the Excel or CSV file.",
        }
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        # Strip any prefix message from DownloadTaskFileTool
        if "File saved to:" in file_path:
            file_path = file_path.split("File saved to:")[-1].split("(")[0].strip()

        if not os.path.exists(file_path):
            return f"[ExcelReadTool ERROR] File not found: {file_path}"

        ext = os.path.splitext(file_path)[-1].lower()

        try:
            import pandas as pd

            if ext == ".csv":
                df_dict = {"Sheet1": pd.read_csv(file_path)}
            elif ext in (".xlsx", ".xls"):
                df_dict = pd.read_excel(file_path, sheet_name=None)
            else:
                # Try CSV as fallback
                try:
                    df_dict = {"Sheet1": pd.read_csv(file_path)}
                except Exception:
                    return f"[ExcelReadTool ERROR] Unsupported file type: {ext}"

            output_parts = []
            for sheet_name, df in df_dict.items():
                output_parts.append(f"=== Sheet: {sheet_name} ===")
                output_parts.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                output_parts.append(df.to_string(index=True, max_rows=200))
                output_parts.append("")

            return "\n".join(output_parts)

        except Exception as exc:
            return f"[ExcelReadTool ERROR] {exc}"
