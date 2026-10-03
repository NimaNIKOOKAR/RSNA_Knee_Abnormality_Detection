from IPython.display import display
import pydicom
import pandas as pd
from matplotlib import pyplot as plt

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