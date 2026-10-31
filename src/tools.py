import SimpleITK as sitk
import numpy as np

def get_mri_volume(series_path : str):
    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(series_path)
    )

    if not files:
        raise ValueError("No DICOM slices found")

    reader.SetFileNames(files)
    image = reader.Execute()

    
    volume = sitk.GetArrayFromImage(image)

    return image, volume.astype(np.float32)
