import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
import os
import glob


def nifti_to_dcm_with_metadata(nifti_file, dicom_dir, output_dir):
    """
    Converts a NIfTI file to a series of DICOM files, using metadata from existing DICOM files.

    Args:
        nifti_file (str): Path to the input NIfTI file.
        dicom_dir (str): Path to the directory containing the original DICOM files.
        output_dir (str): Path to the directory where DICOM files will be saved.
    """
    try:
        # 1. Read the NIfTI file
        nifti_img = nib.load(nifti_file)
        image_data = nifti_img.get_fdata()
        header = nifti_img.header

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get image dimensions
        height, width, num_slices = image_data.shape[:3]

        # 2. Read DICOM metadata from the first file in the directory
        dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in '{dicom_dir}'")

        min_slice_num = np.inf

        for i in range(num_slices):
            dcm = pydicom.dcmread(dicom_files[i])
            min_slice_num = np.min([min_slice_num, dcm.InstanceNumber])

        if min_slice_num > 1:
            int(min_slice_num)
        else:
            min_slice_num = 1

        print(int(min_slice_num))


        # 3. Iterate through slices and create new DICOM files
        for i in range(num_slices):
            # Create a new DICOM dataset
            dcm = pydicom.dcmread(dicom_files[i])
            dcm.ImageComments = "Masked CT Slice"  # Add a comment

            dcm.SamplesPerPixel = 1
            dcm.BitsStored = 16
            dcm.BitsAllocated = 16
            dcm.HighBit = 15
            dcm.PixelRepresentation = 1

            slice_data = image_data[:, :, num_slices - 1 - int(dcm.InstanceNumber - min_slice_num)].astype(np.int16)  # Adjust data type if needed
            slice_data = np.rot90(slice_data,1)

            result_array = dcm.pixel_array
            result_array[slice_data==0] = -1024

            dcm.PixelData = result_array.tobytes()

            # 4. Save the DICOM file
            output_filename = os.path.join(output_dir, f"slice_{i + 1:04d}.dcm")
            file_dataset = FileDataset(output_filename, dcm)
            file_dataset.save_as(output_filename)

        print(
            f"Successfully converted '{nifti_file}' to DICOM files in '{output_dir}' using metadata from '{dicom_dir}'")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    output_vesselFM_dir = './output_vesselFM'  # Replace with your NIfTI directory path
    dicom_directory = "./dicom_original"  # Replace with the DICOM directory of original series
    list_series = os.listdir(dicom_directory)
    masked_dicom_dir = "./dicom_masked"
    os.makedirs(masked_dicom_dir, exist_ok=True)
    for series in list_series:
        nifti_file_path = os.path.join(output_vesselFM_dir, f"{series}_pred.nii.gz")
        output_directory = os.path.join(masked_dicom_dir, series)
        series_dir = os.path.join(dicom_directory, series)
        nifti_to_dcm_with_metadata(nifti_file_path, series_dir, output_directory)
