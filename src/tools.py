from IPython.display import display
import pydicom
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

def show_dicom_info(dicom_path: str):
    """
    Displays the DICOM metadata for a given DICOM file.

    Parameters:
    dicom_path (str): The path to the DICOM file.
    """


    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Convert the DICOM metadata to a dictionary
    dicom_dict = {elem.keyword: elem.value for elem in dicom_data.iterall() if elem.keyword}

    # Create a DataFrame for better visualization
    dicom_df = pd.DataFrame(list(dicom_dict.items()), columns=['Attribute', 'Value'])

    # Display the DataFrame
    display(dicom_df)

def show_dicom_image(dicom_path: str):
    """
    Displays the image from a given DICOM file.

    Parameters:
    dicom_path (str): The path to the DICOM file.
    """

    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Extract the pixel array
    pixel_array = dicom_data.pixel_array

    # Display the image
    plt.imshow(pixel_array, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()


def explore_repository(root_path):
    root = Path(root_path)

    if not root.is_dir():
        raise NotADirectoryError(f"Invalid directory: {root}")

    print(f"{root.name}/")

    def explore(directory, prefix=""):
        items = sorted(
            directory.iterdir(),
            key=lambda x: (x.is_file(), x.name.lower())
        )

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "

            print(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")

            if item.is_dir() and not item.is_symlink():
                extension = "    " if is_last else "│   "
                explore(item, prefix + extension)

    explore(root)

def index_dicom_repository(root_path):

    root = Path(root_path)
    records = []

    for study_dir in sorted(root.iterdir()):

        if not study_dir.is_dir():
            continue

        for series_dir in sorted(study_dir.iterdir()):

            if not series_dir.is_dir():
                continue

            for dicom_file in series_dir.glob("*.dcm"):

                ds = pydicom.dcmread(
                    dicom_file,
                    stop_before_pixels=True
                )

                records.append({
                    "study_id": study_dir.name,
                    "series_id": series_dir.name,
                    "instance_id": dicom_file.stem,
                    "instance_number": ds.get("InstanceNumber"),
                    "modality": ds.get("Modality"),
                    "rows": ds.get("Rows"),
                    "columns": ds.get("Columns"),
                    "file_path": str(dicom_file)
                })

    return pd.DataFrame(records)