import os
import shutil
import tempfile
import dicom2nifti


def convert_dicom_to_nifti(dicom_input_dir, nifti_output_dir):
    """
    Converts all DICOM eries from a root directory into NIfTI format,
    renames them based on their original folder name, and saves them
    in a single output directory.

    Args:
        dicom_input_dir (str): The root directory containing DICOM series,
                               each in its own sub-folder.
        nifti_output_dir (str): The destination directory for the renamed
                                .nii.gz files.
    """
    # Ensure the final output directory exists
    os.makedirs(nifti_output_dir, exist_ok=True)
    print(f"NIfTI files will be saved in: {nifti_output_dir}")

    list_series = os.listdir(dicom_input_dir)
    for series_folder in list_series:
        input_path = os.path.join(dicom_input_dir, series_folder)

        # Skip if the path is not a directory
        if not os.path.isdir(input_path):
            continue

        # Use a temporary directory to store the initial conversion output
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert the DICOM series to NIfTI in the temporary directory
                dicom2nifti.convert_directory(input_path, temp_dir, compression=True, reorient=True)

                # Get the name of the converted file (usually there's only one)
                nifti_files = os.listdir(temp_dir)
                if not nifti_files:
                    print(f"Warning: No NIfTI file was created for series: {series_folder}")
                    continue

                original_nifti_name = nifti_files[0]
                old_path = os.path.join(temp_dir, original_nifti_name)

                # Define the new, meaningful file name and path
                new_file_name = f"{series_folder}.nii.gz"
                new_path = os.path.join(nifti_output_dir, new_file_name)

                # Move the file from the temp directory to the final destination with the new name
                shutil.move(old_path, new_path)
                print(f"Successfully converted and renamed: {series_folder} -> {new_file_name}")

            except Exception as e:
                print(f"Error converting series {series_folder}: {e}")


if __name__ == "__main__":
    # The directory containing sub-folders of DICOM series
    input_directory = "./dicom_original"

    # The single directory where all final NIfTI files will be saved
    output_directory = "./input_vesselFM"

    convert_dicom_to_nifti(input_directory, output_directory)