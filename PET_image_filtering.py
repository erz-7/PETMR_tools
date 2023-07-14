# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage import filters
import glob
import copy
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import os
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma
import functions as f 

# %%
#define importdir and write images from importdir to targetdir
dir_decision = input("Enter the directory where the dicom files to be filtered are located: ")
dir_decision = dir_decision.replace("\\", "/")

if dir_decision == "test":
    #hardcoded test directory
    collected_input_files = glob.glob("" + "/*.dcm")
    collected_gt = ""
    dir_gt = ""

elif dir_decision != "test":
    if os.path.isdir(dir_decision):
        collected_input_files = glob.glob(dir_decision + "/*.dcm")
    else:
        raise ValueError("No dicom files in the input directory provided/directory does not exist. Please provide a valid directory.")
    #query for GT directory
    dir_gt = input("Enter the directory where the GT dicom files are located. If no GT is available, leave blank: ")
    dir_gt = dir_gt.replace("\\", "/")
    if os.path.isdir(dir_gt):
        collected_gt = glob.glob(dir_gt + "/*.dcm")
    else:
        collected_gt = ""

#LEAVE DIR_GT BLANK TO GENERATE A GT DATASET FROM THE TESTING SET

if len(collected_input_files) > 0:
    print("Number of images in dir_1: " + str(len(collected_input_files)))
else:
    raise ValueError("No dicom files in the input directory provided. Please provide a valid directory.")
if len(collected_gt) > 0:
    print("Number of images in dir_gt: " + str(len(collected_gt)))
else:
    print("No GT directory provided. GT will be generated from the input dataset.")

# %%
#execution block for function
dict_images, dict_z_coord, dict_raw_dicom_data, img_columns, img_rows, stored_resx, stored_resy, sorted_times, dict_xyz = f.grab_images_and_data(collected_input_files, "AcquisitionTime")

# %%
#prompt user on whether they want to estimate sigma, or use a pre-determined sigma and filter images with it
valid = False
while valid == False:
    print("How would you like to proceed with filtering?: (determine_sigma/filter_images)")
    print("filter_images: use pre-determined sigma to obtain filtered images and write to file.")
    print("determine_sigma: brute force method to find optimal sigma for each image. Does not filter images/write to directory.")
    sig_inp = input()
    if sig_inp == "determine_sigma" or sig_inp == "filter_images":
        valid = True
    else:
        print("Invalid input. Please try again.")

# %%
#execution block for dataset truncation
if sig_inp == "determine_sigma":
    valid = False
    while valid == False:
        trunc_decision = input("Would you like to truncate the dataset's slices to calculate sigma? (y/n): ")
        if trunc_decision == "y":
            trunc_decision = True
            valid = True
        elif trunc_decision == "n":
            trunc_decision = False
            valid = True
        else:
            print("Please enter a valid response (y/n).")

    if trunc_decision == True:
        dict_images_trunc = {}
        dict_z_coord_trunc = {}
        dict_raw_dicom_data_trunc = {}

        print("Current number of images per timepoint: " + str(len(dict_images[sorted_times[0]])))
        chosen_range = input("Enter the range of images to be truncated (e.g. 0-10): ")
        chosen_range = chosen_range.split("-")
        for entry in chosen_range:
            entry = int(entry.strip().strip("'"))

        #slice the dictionnaries
        for time in sorted_times:
            temp_images = (dict_images[time])[int(chosen_range[0]):int(chosen_range[1])]
            temp_z = (dict_z_coord[time])[int(chosen_range[0]):int(chosen_range[1])]
            temp_dicom = (dict_raw_dicom_data[time])[int(chosen_range[0]):int(chosen_range[1])]
            #reassign slices
            dict_images_trunc[time] = temp_images
            dict_z_coord_trunc[time] = temp_z
            dict_raw_dicom_data_trunc[time] = temp_dicom
        
        print("Truncation complete. Dataset shrunk from " + str(len(dict_images[sorted_times[0]])) + " images per timepoint to " + 
            str(len(dict_images_trunc[sorted_times[0]])) + " images per timepoint.")

        #overwrite the original dictionaries for ease of access
        dict_images = dict_images_trunc
        dict_z_coord = dict_z_coord_trunc
        dict_raw_dicom_data = dict_raw_dicom_data_trunc
        

# %%
if sig_inp == "determine_sigma":
    valid = False
    while valid == False:
        trunc_decision2 = input("Would you also like to truncate the dataset's timepoints? (y/n): ")
        if trunc_decision2 == "y":
            trunc_decision2 = True
            valid = True
        elif trunc_decision2 == "n":
            trunc_decision2 = False
            valid = True
        else:
            print("Please enter a valid response (y/n).")

    if trunc_decision2 == True:
        print("Current number of timepoints: " + str(len(dict_images)))
        chosen_range = input("Enter the range of images to be truncated (e.g. 0-10): ")
        chosen_range = chosen_range.split("-")
        for entry in chosen_range:
            entry = int(entry.strip().strip("'"))
        chosen_range = np.arange(int(chosen_range[0]), int(chosen_range[1]), 1)

        #slice the dictionnaries
        temp_images = {}
        temp_z = {}
        temp_dicom = {}

        for i in range(len(dict_images)):
            if i in chosen_range:
                temp_images[sorted_times[i]] = dict_images[sorted_times[i]]
                temp_z[sorted_times[i]] = dict_z_coord[sorted_times[i]]
                temp_dicom[sorted_times[i]] = dict_raw_dicom_data[sorted_times[i]]
        
        print("Truncation complete. Dataset shrunk from " + str(len(dict_images)) + " timepoints to " + 
                str(len(temp_images)) + " timepoints.")
        
        #overwrite the original dictionaries for ease of access
        dict_images = temp_images
        dict_z_coord = temp_z
        dict_raw_dicom_data = temp_dicom

        sorted_times = sorted_times[chosen_range[0]:chosen_range[-1]]

# %%
#if ground truth dataset is provided, get ground truth images and sort them by z-coordinate
if collected_gt != "":
    gt_images, gt_z_coord, gt_raw_dicom_data, gt_columns, gt_rows, gt_resx, gt_resy, gt_times = f.grab_images_and_data(collected_gt, "AcquisitionTime")
  
    #gt DICOM metadata is not required, set to None to save memory
    gt_raw_dicom_data = None

    assert gt_columns == img_columns and gt_rows == img_rows, "Inconsistent image resolution found between ground truth and images. Please check and reformat your data."
    print("Number of images in ground truth dataset:", len(gt_images))

    #repeat ground truth images for each timepoint in target dataset
    gt_images_init = []
    gt_images_repeated = []

    for i in range (len(sorted_times)):
        gt_images_init.append(gt_images)

    #going one short for some weird reason, rough fix
    gt_images_init.append(gt_images)

    for sublist in gt_images_init:
        for item in sublist:
            gt_images_repeated.append(item)

    assert gt_z_coord[0] == dict_z_coord[0] and gt_z_coord[-1] == dict_z_coord[-1], "Z-coordinates of ground truth images do not match z-coordinates of images in dict_images."

else:
    #create synthetic ground truth dataset from averaged means of all images in target dataset
    #only if no ground truth dataset is provided
    
    def create_synthetic_gt(input_dataset_dict, timepoints):
        """Creates synthetic ground truth dataset from averaged means of all images in target dataset. Extends this ground truth dataset 
        to match the number of images in the target dataset."""
        
        gt_images_init = []

        #average all images in each key in dict_images to create synthetic ground truth images
        for i in range (len(input_dataset_dict[timepoints[0]])):
            imgs_temp = []
            for z in range (len(timepoints)):
                imgs_temp.append(input_dataset_dict[timepoints[z]][i])
            gt_images_init.append(np.mean(imgs_temp, axis=0))

        #repeat ground truth images to match number of images in each key in dict_images. Just make it a list to match the format of all_img
        gt_images_1timepoint = []
        gt_images_ext = []

        for i in range (len(timepoints)):
            global sig_inp
            if sig_inp == "filter_images":
                gt_images_1timepoint.append(gt_images_init)

        #going one short for some weird reason, rough fix -- EDIT: no longer needed? weird.
        #gt_images_1timepoint.append(gt_images_init)

        for sublist in gt_images_1timepoint:
            for item in sublist:
                gt_images_ext.append(item)

        print("Synthetic GT dataset created, with", len(gt_images_ext), "images.")
        return gt_images_ext
    
    gt_images_repeated = create_synthetic_gt(dict_images, sorted_times)

# %%
#simplify things by transforming image dictionary into one list for processing
assert len(dict_images[sorted_times[-1]]) == len(dict_images[sorted_times[0]])

all_img = []

for key in dict_images:
    for i in range (len(dict_images[key])):
        all_img.append(dict_images[key][i])

all_dicom = []

for key in dict_raw_dicom_data:
    for i in range (len(dict_raw_dicom_data[key])):
        all_dicom.append(dict_raw_dicom_data[key][i])

#create a deepcopy of all_img to be used for filtering
all_img_filt = copy.deepcopy(all_img)

assert len(all_img) == len(gt_images_repeated), "Number of images in all_img and synthetic GT repeat list do not match."

# %%
#plot first 5 original images and first 5 gaussian filtered images in a subplot to check copy
#add ground truth images to the mix

fig, ax = plt.subplots(3, 5, figsize = (20, 4))

for i in range (5):
    ax[0, i].imshow(all_img[i], cmap = "gray")
    ax[1, i].imshow(all_img_filt[i], cmap = "gray")
    ax[2, i].imshow(gt_images_repeated[i], cmap = "gray")

# %%
    
if sig_inp == "filter_images":
    #sigma estimation

    #ask user how sigma should be estimated
    valid = False
    while valid == False:
        sig_method = input("Please choose a method for estimating sigma (manual, auto, auto_mean): ")
        if sig_method == "manual" or sig_method == "auto" or sig_method == "auto_mean":
            valid = True
        else:
            print("Invalid input. Please try again.")

    #execution block for sigma prompts
    #get sigma automatically using the estimate_sigma function from sklearn
    if sig_method == "auto":
        print("Estimating sigma...")
        sigmas = []
        for i in tqdm(range(len(all_img))):
            sigma_temp = (np.mean(estimate_sigma(all_img[i], channel_axis=None)))
            sigma_conv = sigma_temp.tolist()
            sigmas.append(sigma_conv)

    #same as above, use sklearn to estimate sigma but take the mean of all sigma values for the dataset
    elif sig_method == "auto_mean":
        print("Estimating sigma...")
        sigmas = []
        for i in tqdm(range(len(all_img_filt))):
            sigma_temp = (np.mean(estimate_sigma(all_img_filt[i], channel_axis=None)))
            sigma_conv = sigma_temp.tolist()
            sigmas.append(sigma_conv)
        sigma_float = np.mean(sigmas)
        print("Mean sigma value is:", sigma_float)
        sigmas = [sigma_float for i in range(len(all_img_filt))]

    #manually enter sigma value - if user has done a "brute force" check beforehand (recommended)
    elif sig_method == "manual":
        valid = False
        while valid == False:
            sigma = input("Please enter a sigma value: ")
            try:
                sigma_man = float(sigma)
                valid = True
            except:
                print("Invalid input. Please try again.")
        sigmas = [sigma_man for i in range(len(all_img))]
        print("Chosen sigma value is:", sigma)

    for sig in sigmas:
        if sig == np.nan:
            print("NAN VALUE FOUND - sigma assigned backup value of 1.0")
            sig = 1

# %%
#prompt user on whether they want to filter images or not
valid = False
while valid == False:
    filt = input("Please choose a filter for denoising (gauss, NLM, wavelet, none): ")
    if filt == "gauss" or filt == "NLM" or filt == "wavelet" or filt == "none":
        valid = True
    else:
        print("Invalid input. Please try again.")

print("Image filter chosen:", filt)

#if user chooses NLM filter, prompt them on whether they want to use a pre-determined kernel (norm) or not (deep)
if filt == "NLM" and sig_inp == "determine_sigma":
    valid = False
    while valid == False:
        print("Please choose a seatch depth for brute NLM (norm, deep): ")
        searchdepth = input("norm uses a hard-coded kernel size for processing, while deep iterates through several possible kernel sizes at once")
        if searchdepth == "norm" or searchdepth == "deep":
            valid = True
        else:
            print("Invalid input. Please try again.")

# %%
if sig_inp == "filter_images":

    #reset all_img_filt in case variable resets are needed
    all_img_filt = copy.deepcopy(all_img)

    if filt == "none":
        print("No filter applied.")
        #do nothing, run this to check SNR and SSIM between GT and images

    if filt == "gauss":

        #apply filter and track progress
        print("Applying gaussian filter...")
        for i in tqdm(range(len(all_img_filt))):
            all_img_filt[i] = filters.gaussian(all_img_filt[i], sigma = sigmas[i], preserve_range = True)

    if filt == "wavelet":

        #apply filter and track progress
        print("Applying wavelet filter...")
        for i in tqdm(range(len(all_img_filt))):
            all_img_filt[i] = denoise_wavelet(all_img_filt[i], sigma = sigmas[i], channel_axis = None, method = "BayesShrink",
                                                mode = "soft", wavelet = "bior4.4", rescale_sigma = True)

    if filt == "NLM":
            
        #define patch parameters: 3x3 patchsize, 11x11 search area
        patch_kw = dict(patch_size=3,      # 3x3 patches
                        patch_distance=8,  # 11x11 search area
                        channel_axis=None,)
            
        #query user for fast mode
        valid = False
        while valid == False:
            speed = input("Would you like to use fast mode? (y/n): ")
            if speed == "y" or speed == "n":
                valid = True
            else:
                print("Invalid input. Please try again.")
            if speed == "y":
                speed = True
            elif speed == "n":
                speed = False

        #apply filter and track progress
        print("Applying non-local means filter...")
        for i in tqdm(range(len(all_img_filt))):
            all_img_filt[i] = denoise_nl_means(all_img_filt[i], h = 0.8 * sigmas[i], sigma = sigmas[i], fast_mode = speed, **patch_kw)


##############################################################################################################

if sig_inp == "determine_sigma":


    sig_range = input("Please enter a range of sigma values to test as well as a step value to use (e.g. 0.1,3.0,0.1): ")
    sig_range = sig_range.split(",")
    print("Sigma range is:", sig_range[0], "to", sig_range[1], "with a step size of", sig_range[2])
    sig_range = [float(i) for i in sig_range]
    sig_range_def = np.arange(sig_range[0], sig_range[1], sig_range[2], dtype=float)

    sig_results = []

    if filt == "none" or filt == "gauss" or filt == "wavelet" or (filt == "NLM" and searchdepth == "norm"):

        if filt == "none":
            print("Brute force sigma optimization without a filter? You're somewhat of a funny individual, you know?")
            #do nothing, run this to check SNR and SSIM between GT and images

        if filt == "gauss":
            
            print("Applying gaussian filter...")
            mean_psnr = []
            mean_ssim = []

            #iterate through entire sigma range to check which sigma optimizes parameters
            for i in tqdm(range(len(sig_range_def))):

                #reset all_img_filt in case variable resets are needed at each sigma
                all_img_filt = copy.deepcopy(all_img)

                #apply filter and track progress
                curr_psnr = []
                curr_ssim = []

                #apply gaussian filter to all images and get psnr and ssim values for each image
                for z in range(len(all_img_filt)):
                    all_img_filt[z] = filters.gaussian(all_img_filt[z], sigma = sig_range_def[i], preserve_range = True)
                    curr_psnr_calc = psnr(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0)
                    if curr_psnr_calc != np.inf and curr_psnr_calc != -np.inf:
                        curr_psnr.append(curr_psnr_calc)
                    curr_ssim.append(ssim(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0))
            
                mean_psnr.append(np.nanmean(curr_psnr))
                mean_ssim.append(np.nanmean(curr_ssim))

        if filt == "wavelet":

            print("applying wavelet filter...")
            mean_psnr = []
            mean_ssim = []

            #iterate through entire sigma range to check which sigma optimizes parameters
            for i in tqdm(range(len(sig_range_def))):
                
                #reset all_img_filt in case variable resets are needed at each sigma
                all_img_filt = copy.deepcopy(all_img)

                curr_psnr = []
                curr_ssim = []

                #apply wavelet filter to all images and get psnr and ssim values for each image
                for z in range(len(all_img_filt)):
                    all_img_filt[z] = denoise_wavelet(all_img_filt[z], sigma = sig_range_def[i], channel_axis = None, method = "BayesShrink",
                                                    mode = "soft", wavelet = "db1", rescale_sigma = True)
                    curr_psnr_calc = psnr(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0)
                    if curr_psnr_calc != np.inf and curr_psnr_calc != -np.inf:
                        curr_psnr.append(curr_psnr_calc)
                    curr_ssim.append(ssim(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0))
                
                #get mean psnr and ssim values for each sigma value
                mean_psnr.append(np.nanmean(curr_psnr))
                mean_ssim.append(np.nanmean(curr_ssim))

        if filt == "NLM" and searchdepth == "norm":
            
            print("applying non-local means filter...")
            mean_psnr = []
            mean_ssim = []

            #iterate through entire sigma range to check which sigma optimizes parameters
            for i in tqdm(range(len(sig_range_def))):
                
                #reset all_img_filt in case variable resets are needed at each sigma
                all_img_filt = copy.deepcopy(all_img)

                curr_psnr = []
                curr_ssim = []

                #apply wavelet filter to all images and get psnr and ssim values for each image
                for z in range(len(all_img_filt)):
                    all_img_filt[z] = denoise_nl_means(all_img_filt[z], h = 0.8 * sig_range_def[i], sigma = sig_range_def[i], fast_mode = True, 
                                     patch_size = 5, patch_distance = 11, channel_axis = None)
                    curr_psnr_calc = psnr(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0)
                    if curr_psnr_calc != np.inf and curr_psnr_calc != -np.inf:
                        curr_psnr.append(curr_psnr_calc)
                    curr_ssim.append(ssim(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0))
                
                #get mean psnr and ssim values for each sigma value
                mean_psnr.append(np.nanmean(curr_psnr))
                mean_ssim.append(np.nanmean(curr_ssim))

        #plot ssim and psnr scores versus sigma in two subplots

        fig, ax = plt.subplots(1, 2, figsize = (20, 8))

        ax[0].plot(sig_range_def, mean_psnr, color = "red")
        ax[0].set_title("PSNR vs. Sigma")
        ax[0].set_xlabel("Sigma")
        ax[0].set_ylabel("PSNR")

        ax[1].plot(sig_range_def, mean_ssim, color = "green")
        ax[1].set_title("SSIM vs. Sigma")
        ax[1].set_xlabel("Sigma")
        ax[1].set_ylabel("SSIM")

        #report exaclty when psnr and ssim are optimized per sigma value
        print("PSNR and SSIM plots generated. PSNR optimized at sigma =", sig_range_def[np.argmax(mean_psnr)], "with a value of", np.max(mean_psnr))
        print("SSIM optimized at sigma =", sig_range_def[np.argmax(mean_ssim)], "with a value of", np.max(mean_ssim))

    if filt == "NLM" and searchdepth == "deep":
        
        print("applying non-local means filter...")

        kernel_range = [3, 4, 5, 6, 7]
        cont_psnr = []
        cont_ssim = []

        counter = 0
        counter_imgproc = 0

        for i in tqdm(range(len(kernel_range))):

            kernel_psnr = []
            kernel_ssim = []

            for y in range (len(sig_range_def)):

                #reset all_img_filt in case variable resets are needed at each sigma
                all_img_filt = copy.deepcopy(all_img)

                curr_psnr = []
                curr_ssim = []

                #apply NLM filter to all images and get psnr and ssim values for each image
                for z in range(len(all_img_filt)):
                    all_img_filt[z] = denoise_nl_means(all_img_filt[z], h = 0.8 * sig_range_def[y], sigma = sig_range_def[y], fast_mode = True, 
                                     patch_size = kernel_range[i], patch_distance = 11, channel_axis = None)
                    counter_imgproc += 1
                    curr_psnr_calc = psnr(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0)
                    if curr_psnr_calc != np.inf and curr_psnr_calc != -np.inf:
                        curr_psnr.append(curr_psnr_calc)
                    else:
                        counter += 1
                    curr_ssim.append(ssim(gt_images_repeated[z], all_img_filt[z], data_range = 1275.0))
                
                #get mean psnr and ssim values for each sigma value or nanmean of current list of psnr/sigma values
                kernel_psnr.append(np.nanmean(curr_psnr))
                kernel_ssim.append(np.nanmean(curr_ssim))

            cont_psnr.append(kernel_psnr)
            cont_ssim.append(kernel_ssim)

        #report on the amount of inf values that were generated and the amount of images processed for diagnostic purposes
        print("Number of PSNR values that were inf:", counter)
        print("Number of images processed:", counter_imgproc)


# %%
if filt == "NLM" and sig_inp == "determine_sigma":
      #plot psnr scores versus sigma for each kernel value in a gradient surface 3D subplot

        fig = plt.figure(figsize = (30, 12))
        ax = fig.add_subplot(1, 2, 1, projection = "3d")


        X1, Y1 = np.meshgrid(sig_range_def, kernel_range)
        Z1 = np.array(cont_psnr).reshape(X1.shape)

        surf = ax.plot_surface(X1, Y1, Z1, cmap = plt.get_cmap("winter"))
        fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
        ax.set_title("PSNR vs. Sigma vs. Kernel Size")
        ax.set_xlabel("Sigma")
        ax.set_ylabel("Kernel Size")
        ax.set_zlabel("PSNR")

        plt.show()
        plt.close()

# %%
if filt == "NLM" and sig_inp == "determine_sigma":
      #plot ssim scores versus sigma for each kernel value in a gradient surface 3D subplot

        fig = plt.figure(figsize = (30, 12))
        ax = fig.add_subplot(1, 2, 1, projection = "3d")


        X1, Y1 = np.meshgrid(sig_range_def, kernel_range)
        Z1 = np.array(cont_ssim).reshape(X1.shape)

        surf = ax.plot_surface(X1, Y1, Z1, cmap = plt.get_cmap("hot"))
        fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
        ax.set_title("SSIM vs. Sigma vs. Kernel Size")
        ax.set_xlabel("Sigma")
        ax.set_ylabel("Kernel Size")
        ax.set_zlabel("SSIM")

        plt.show()
        plt.close()

# %%
#plot first 5 original images and first 5 gaussian filtered images in a subplot to check if filtering functioned properly
#can also check appearance of filtered images in the "all_img_filt" list

fig, ax = plt.subplots(2, 5, figsize = (20,8))

for i in range (5):
    ax[0, i].imshow(all_img[i], cmap = "gray")
    ax[1, i].imshow(all_img_filt[i], cmap = "gray")

# %%
#get metrics for images in proc_images relative to their unprocessed counterparts
if sig_inp == "filter_images":

    valid = False
    while valid == False:
        print("Would you like to define a range of slices for metric calculation? (y/n): ")
        dec = input("Useful if you know where tumour is in slices/want to exclude machine noise outliers")
        if dec == "y":
            valid = True
            range_def = True
        elif dec == "n":
            valid = True
            range_def = False
        else:
            print("Invalid input. Please try again.")

    ssim_scores = []
    psnr_scores = []
    maxes = []
    mins = []

    #typecast all images to float, get max and min values for each image to get intensity range
    for i in range (len(all_img_filt)):
        all_img_filt[i] = all_img_filt[i].astype(float)
    for z in range (len(all_img)):
        all_img[i] = all_img[i].astype(float)
        maxes.append(all_img[i].max())
        mins.append(all_img[i].min())

    max = np.max(maxes)
    min = np.min(mins)

    print("pixel intensity range - max:", max, "min:", min)
    range_est = max - min

    #get metrics for whole filtered dataset if user does not specify which image slices to get metrics for
    if range_def == False:
        print("Getting metrics for whole dataset...")
        for i in range(len(all_img_filt)):
            ssim_scores.append(ssim(gt_images_repeated[i], all_img_filt[i], data_range = range_est))
            psnr_calc = psnr(gt_images_repeated[i], all_img_filt[i], data_range = range_est)
            #remove data outliers
            if psnr_calc != np.inf and psnr_calc != -np.inf:
                psnr_scores.append(psnr_calc)
    
    #get metrics for specified range of slices if user specifies which image slices to get metrics for
    elif range_def == True:
        print("Amount of slices per timepoint processed:", len(dict_images[sorted_times[0]]))
        
        #prompt user on which slices to get metrics for
        valid = False
        while valid == False:
            print("Current range of slices per timepoint is 0 -", len(dict_images[sorted_times[0]]) - 1, "inclusive.")
            spec_range = input("Please enter a range of slices to calculate metrics for (e.g. 0-5): ")
            spec_range = spec_range.split("-")
            spec_range = [int(i) for i in spec_range]
            valid = True
               
        #extend range to include all slices for each timepoint - e.g. if user specifies 0-5, range will be 0-5, 6-11, 12-17 etc.
        #as above but on a slace where each timepoint has 100+ slices
        iters = []
        for i in range(len(dict_images)):
            iters.append(i)

        print("Slices excluded: " + "0 -" + str(spec_range[0]) + " - " + str(len(dict_images[sorted_times[0]]) + "-" + str(spec_range[1])))

        len_1_iter = len(dict_images[sorted_times[0]])
        total_len = (len(iters)) * len_1_iter
        cont = []

        #extend user's choices for a single timepoint to all timepoints until slices are out of range
        #get all ranges within dataset where metrics will be calculated
        for i in range(len(iters)):
            addition = iters[i]*len_1_iter
            if spec_range[1] + addition <= total_len:
                cont.append([spec_range[0] + addition, spec_range[1] + addition])
              
        #use numpy arange to get all slices within the ranges specified above
        #final slice list is specified by cont_final_slices
        cont_final_slices = []
        for i in range(len(cont)):
            cont_final_slices.append((np.arange(cont[i][0], cont[i][1], 1)))

        cont_final_slices = np.array(cont_final_slices)
        cont_final_slices.ravel()
        print("Done excluding slices. Calculating metrics...")

        #get metrics for all slices in cont_final_slices
        for i in range(len(all_img_filt)):
            if i in cont_final_slices:
                ssim_scores.append(ssim(gt_images_repeated[i], all_img_filt[i], data_range = range_est))
                psnr_calc = psnr(gt_images_repeated[i], all_img_filt[i], data_range = range_est)
                #remove data outliers
                if psnr_calc != np.inf and psnr_calc != -np.inf:
                    psnr_scores.append(psnr_calc)

    print("Done calculating metrics. Plotting...")
    #plot ssim and psnr scores in two subplots
    fig, ax = plt.subplots(1, 2, figsize = (20, 8))

    ax[0].plot(ssim_scores, color = "orange", linewidth = 0.5)
    ax[0].set_title("SSIM Scores")
    ax[0].set_xlabel("Image Number")
    ax[0].set_ylabel("SSIM Score (%)")

    ax[1].plot(psnr_scores, color = "blue", linewidth = 0.5)
    ax[1].set_title("PSNR Scores")
    ax[1].set_xlabel("Image Number")
    ax[1].set_ylabel("PSNR Score (dB)")

    valid = False
    while valid == False:
        save = input("Would you like to save the plot? (y/n): ")
        if save == "y":
            valid = True
            savedir_plot = input("Please enter a directory to save the plot to: ")
            if os.path.isdir(savedir_plot) == False:
                os.mkdir(savedir_plot)
            os.chdir(savedir_plot)  
            plt.savefig(savedir_plot + "//" + "psnr_ssim_plot.png")
        elif save == "n":
            print("Plot not saved.")
            valid = True
        else:
            print("Invalid input. Please try again.")
    plt.close(fig)

    #calculate mean ssim and psnr scores
    print("Mean SSIM - " + str(np.nanmean(ssim_scores)))
    print("Mean PSNR - "+ str(np.nanmean(psnr_scores)))

    print("Peak PSNR - " + str(np.nanmax(psnr_scores)))
    print("Peak SSIM - " + str(np.nanmax(ssim_scores)))

# %%
print("Some PET datasets come with inconsistent pixel spacing throughout the dataset due to rounding errors over time")
print("This can cause issues with registration and other processing steps")

valid = False
while valid == False:
    z_corr_decision = input("Would you like this script to correct all image z positions to ensure consistent spacing between slices? (y/n): ")
    if z_corr_decision == "y":
        z_corr_decision = True
        valid = True
    elif z_corr_decision == "n":
        z_corr_decision = False
        valid = True
    else:
        print("Invalid input. Please enter 'y' or 'n'")

if z_corr_decision == True:
    # Z position correciton block
    #get spacing between slices for entire dataset
    slice_spacing = []
    for i in range (len(dict_z_coord[sorted_times[0]])):
        if i == 0:
            continue
        else:
            slice_spacing.append(abs(dict_z_coord[sorted_times[0]][i] - dict_z_coord[sorted_times[0]][i-1]))

    #get actual slice thickness - set type to float64 to match initial dicom z pos
    init = dict_z_coord[sorted_times[0]][0]
    end = dict_z_coord[sorted_times[0]][-1]
    slice_thickness = round((abs(init - end) / (len(dict_z_coord[sorted_times[0]]) - 1)), 8) # set to 8 sig figs to match initial dicom z pos

    #correct z positions for spacing between slices
    z_corr = np.arange(init, end + slice_thickness, slice_thickness)
    z_corr = z_corr.tolist()

    for i in range (len(z_corr)):
        z_corr[i] = round(z_corr[i], 8)

    z_corr = np.array(z_corr)
    z_corr.astype("float64")
    #assert len(z_corr_repeated) == len(all_img_filt) #check that z_corr_repeated is the same length as all_img_filt

    new_xyz = []
    x_init = dict_xyz[sorted_times[0]][0][0]
    y_init = dict_xyz[sorted_times[0]][0][1]

    #get new xyz positions for each slice
    for i in range(len(dict_xyz[sorted_times[0]])):
        new_xyz.append([x_init, y_init, z_corr[i]])

    #extend new xyz positions to all timepoints
    new_xyz_repeated = copy.deepcopy(new_xyz)
    
    for i in range (len(sorted_times) - 1):
        new_xyz_repeated += new_xyz

    print("Done correcting z positions.")
    slice_thickness = np.float64(slice_thickness)

# %%
#if user has chosen to fiilter images, writeblock of code to write images to specified directory in specified file format

def write_filtered_data_to_dir(input_data, program_method, syn_gt, dicom_data):
    """Function takes as input image data and writes it to a specified directory in a specified file format."""

    if program_method == "filter_images":
        valid = False
        while valid == False:
            specified_savedir = input("Enter the directory you want to save the images to. Please do not include any quotes: ")
            if os.path.isdir(specified_savedir):
                valid = True
            else:
                print("Invalid directory/directory does not exist. Please try again.")

        #write images to specified directory in specified file format
        #user is prompted whether to write images in csv format (pixel matrix only) or dcm format (updated pixel matrix and initial metadata)
        valid = False
        while valid == False:
            print("Enter the file format you want to save the images as. Please do not include any quotes.")
            specified_fileformat = input("Choices are: csv, dcm: ")
            if specified_fileformat == "csv" or specified_fileformat == "dcm":
                valid = True
            else:
                print("Invalid file format. Please try again.")

        #prompt the user on what the casename should be called
        valid = False
        while valid == False:
            casename = input("Enter the casename you want to save the images as. Please do not include any quotes: ")
            if casename != "":
                #check for ilegal characters in casename
                if "!" in casename or ">" in casename or "<" in casename or "?" in casename or "/" in casename or "|" in casename or ":" in casename or "*" in casename or '"' in casename:
                    print("Please do not include any of the following characters in your casename: !, >, <, ?, /, |, :, *, .")
                else:
                    valid = True
            else:
                print("Please do not leave this field blank. You'll thank me later!")

        #get some global variables to help with writing process
        global filt
        global dict_images
        global sorted_times

        #write files as csv to specified directory
        if specified_fileformat == "csv":
            counter = 0
            print("Writing to CSV files...")
            for i in range (len(input_data)):
                img_name = casename + "_" + filt + "__" + str(int(i/(len(dict_images[sorted_times[0]])))) + "__" + str(counter) + ".csv"
                savepath = specified_savedir + "\\" + img_name
                np.savetxt(savepath, input_data[i], delimiter=",")
                counter += 1
                if counter == len(dict_images[sorted_times[0]]):
                    counter = 0

        #write files as dcm to specified directory
        if specified_fileformat == "dcm":
            counter = 0
            print("Writing to DICOM files...")
            for i in tqdm(range(len(input_data))):
                img_name = casename + "_" + filt + "_" + str(int(i/(len(dict_images[sorted_times[0]])))) + "__" + str(counter) + ".dcm"
                savepath = specified_savedir + "\\" + img_name
                dcm = dicom_data[i]
                #specify some dicom metadata parameters/typecast pixel array to ensure images are written properly 
                dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dcm.is_little_endian = True
                dcm.is_implicit_VR = False
                dcm.PixelData = input_data[i].astype(np.uint16).tobytes()
                if z_corr_decision == True:
                    dcm.ImagePositionPatient = new_xyz_repeated[i]
                    dcm.SliceThickness = slice_thickness
                dcm.save_as(savepath, write_like_original=True)
                counter += 1
                if counter == len(dict_images[sorted_times[0]]):
                    counter = 0

            #query user if they want to convert the saved dicom files to nifti files as well
            valid = False
            while valid == False:
                usr_inp = input("Would you also like to convert the saved DICOM files to the NIFTI format? (y/n)")
                if usr_inp == "y":
                    valid = True
                    #convert all saved dicom files to nifti
                    f.convert_dcm_to_nii(specified_savedir, specified_savedir + "\\nifti", casename)

                    print("DICOM files converted to NIFTI format and saved in", specified_savedir + "\\nifti")
                elif usr_inp == "n":
                    valid = True
                else:
                    print("Invalid input. Please try again.")


        print("Complete. All files written in", specified_fileformat, "format images to specified directory", specified_savedir)

    #if user has chosen to estimate sigma only, do not write anything
    if program_method == "determine_sigma":
        print("Nothing written, brute is for sigma optimization only. Please re-run using algorithmic sigma method and input ideal sigma")

    #if user has chosen to create and use synthetic ground truth, write images to specified directory in specified file format
    global dir_gt

    if dir_gt == "":
        valid = False
        while valid == False:
            decs = input("Synthetic ground truth was created in the current script. Would you like to save it for later reference/use? (y/n)")
            if decs == "y" or decs == "n":
                valid = True
            else:
                print("Invalid input. Please try again.")

        #only CSV format is supported for ground truth images for the moment
        if decs == "y":
            print("Saving ground truth images to same directory as filtered images...")
            counter = 0
            for i in range (len(input_data)):
                img_name = casename + "_GT_" + str(counter) + ".csv"
                savepath = specified_savedir + "\\" + img_name
                np.savetxt(savepath, syn_gt[i], delimiter=",")
                counter += 1

    return

#call functions
valid = False
while valid == False:
    write_decision = input("Would you like to write the filtered images to a specified directory? (y/n): ")
    if write_decision in ["y", "Y"]:
        write_filtered_data_to_dir(all_img_filt, sig_inp, 
                               gt_images_repeated[0:(len((dict_z_coord[sorted_times[0]])))], all_dicom)
        valid = True
    elif write_decision in ["n", "N"]:
        valid = True
        print("Nothing written. Exiting script")
    else:
        print("Please enter a valid response (y/n).")



