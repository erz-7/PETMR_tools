import numpy as np
from tqdm import tqdm
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as py

def grab_images_and_data(input_dir, sort_method):
    """Grab pixel arrays, z positions and raw dicom file imports from input directory. Return them for later use in program, all sorted by
    z position."""

    sort_list = []

    images = {}
    z_coord = {}
    raw_dicom_data = {}
    xyz = {}

    #track which images are sorted where - informational purposes only
    counter = 0

    print("Reading DICOM files and grabbing image data...")
    for i in tqdm(input_dir):

        #read DICOM file
        filename = i.strip()
        dcm = pydicom.read_file(filename)

        #check image validity and acquire image data based on chosen classifier (in sort) for subdirectory of choice
        if "ImageType" in dcm:
            current_sort = dcm[sort_method].value

            if current_sort not in sort_list:
                sort_list.append(current_sort)
                images[current_sort] = []
                z_coord[current_sort] = []
                raw_dicom_data[current_sort] = []
                xyz[current_sort] = []

            #grab all data and place them into current dict key
            images[current_sort].append(dcm.pixel_array)
            z_coord[current_sort].append(dcm.ImagePositionPatient[2])
            raw_dicom_data[current_sort].append(dcm)
            xyz[current_sort].append(dcm.ImagePositionPatient)

            #get image size
            if counter == 0:
                resx = dcm.Columns
                resy = dcm.Rows
                counter += 1

            #check if image size is consistent
            if counter > 0:
                if dcm.Columns != resx or dcm.Rows != resy:
                    raise ValueError("Inconsistent image resolution found. Please check and reformat your data.")

    columns = dcm.Columns
    rows = dcm.Rows

    print("Resolution of images in this dataset is:", columns, "pixels (X) by", rows, "pixels (Y)")
    print("Voxel dimensions =", dcm.PixelSpacing[0], "mm (X) by", dcm.PixelSpacing[1], "mm (Y) by", dcm.SliceThickness, "mm (Z)")
    timepoints = sorted(sort_list)

    #sort all images in each key by z-coordinate
    for i in range (len(sort_list)):
        img_to_sort = images[timepoints[i]]
        z_coord_to_sort = z_coord[timepoints[i]]
        raw_to_sort = raw_dicom_data[timepoints[i]]
        xyz_to_sort = xyz[timepoints[i]]

        sorted_z, sorted_img = sort_list_by_np(z_coord_to_sort, img_to_sort)
        sorted_z, sorted_raw = sort_list_by_np(z_coord_to_sort, raw_to_sort)
        sorted_z, sorted_xyz = sort_list_by_np(z_coord_to_sort, xyz_to_sort)

        #reassign sorted lists to dictionary
        images[timepoints[i]] = list(sorted_img)
        z_coord[timepoints[i]] = list(sorted_z)
        raw_dicom_data[timepoints[i]] = list(sorted_raw)
        xyz[timepoints[i]] = list(sorted_xyz)

    print("Sorted all images by z-coordinate.")
    print(sorted_z.tolist())
    print("Timepoints found in this dataset:")
    print(timepoints)

    return images, z_coord, raw_dicom_data, columns, rows, resx, resy, timepoints, xyz

############################################################################################################################################################################

def sort_list_by_np(list1,list2):
    """Sorts list1 and list2 by list1 using numpy indexing."""
    
    idx = np.argsort(list1)
    sorted_list1 = np.array(list1)[idx]
    sorted_list2 = np.array(list2)[idx]

    return sorted_list1, sorted_list2

############################################################################################################################################################################


def convert_dcm_to_nii(dcm_dir, nii_dir, savename):
    """
    Convert DICOM files to NIfTI files
    :param dcm_dir: directory containing DICOM files
    :param nii_dir: directory to save NIfTI files
    :return: None
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Added a call to PermuteAxes to change the axes of the data
    #image = sitk.PermuteAxes(image, [2, 1, 0])

    sitk.WriteImage(image, nii_dir + "\\" + savename + '.nii.gz')

############################################################################################################################################################################


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])