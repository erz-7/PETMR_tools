# %%
import glob
import os
import functions as f
from tqdm import tqdm
from pydicom.uid import ExplicitVRLittleEndian
import pydicom

# %%
#define file import directories for chosen modality (user input) and grab DICOM images from directory using glob
valid = False

while valid == False:
    search_datapath = input("Enter the path to the directory containing the PET/MRI data: ")
    fetched = len(glob.glob(search_datapath + "/*.dcm")) + len(glob.glob(search_datapath + "/*.IMA"))
    
    if fetched > 0:
        valid = True
        dicom_files = glob.glob(search_datapath + "/*.dcm") + glob.glob(search_datapath + "/*.IMA")
        print("DICOM files found. Proceeding with data import.")
        break

    else:
        print("No DICOM files found in the specified directory. Please try again.")
        continue

# %%
#execution block for series tag grabber
full_search = input("Would you like to see a list of all series tags and patient IDs in the dataset? (y/n): ")
if full_search == "y" or full_search == "Y":
    series, patients = f.grab_series_patients(dicom_files)
#can just use cached tags to save time
else:
    print("Using cached series tags and patient IDs.")
    series = ['t1_vibe_tra_dyn', 'Static 20-45 min_PRR_NAC Images', 'Dynamic 0-20 min_PRR_NAC Images']
    patients = ['FLT-PETMR-003','FLT-PETMR-006']
    print("Different MRI/PET series fetched:"); print(sorted(series))
    print("Patients examined:"); print(sorted(patients))


valid = False
while valid == False:
    chosen_series = input("Enter the series tag of the desired MRI/PET series for study. Please do not include any quotes: ")
    chosen_patient = input("Enter the patient ID of the (single) desired patient for study. Please do not include any quotes: ")

    #check that user input is valid, else crash
    if chosen_series in series and chosen_patient in patients:
        valid = True
        print("Input series and patient ID are valid.")
    else:
        print("Invalid series or patient ID input. Please try again.")

# %%
dict_z_coords, dict_raw_dicoms, timepoints = f.grab_images_and_data_series_patient(dicom_files, "AcquisitionTime", chosen_series, chosen_patient)

list_raw_dicoms = []
list_z = []

#linearize the data
for i in range (len(timepoints)):
    for z in range (len(dict_raw_dicoms[timepoints[i]])):
        list_raw_dicoms.append(dict_raw_dicoms[timepoints[i]][z])

for i in range (len(timepoints)):
    for z in range (len(dict_z_coords[timepoints[i]])):
        list_z.append(dict_z_coords[timepoints[i]][z])

# %%
valid = False
while valid == False:
    specified_savedir = input("Enter the directory you want to save the images to. Please do not include any quotes: ")
    if os.path.isdir(specified_savedir):
        valid = True
    else:
        print("Invalid directory/directory does not exist. Please try again.")

counter = 0
print("Writing to DICOM files...")
for i in tqdm(range(len(list_raw_dicoms))):
    img_name = chosen_patient + "__" + chosen_series + "__" + str(int(i/(len(dict_raw_dicoms[timepoints[0]])))) + "__" + str(counter) + ".dcm"
    savepath = specified_savedir + "\\" + img_name
    dcm = list_raw_dicoms[i]
    #specify some dicom metadata parameters/typecast pixel array to ensure images are written properly 
    dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dcm.is_little_endian = True
    dcm.is_implicit_VR = False
    dcm.save_as(savepath, write_like_original=True)
    counter += 1
    if counter == len(dict_raw_dicoms[timepoints[0]]):
        counter = 0


