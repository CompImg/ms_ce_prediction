import os
import nibabel as nib
import numpy as np
from skimage.morphology import label, remove_small_objects
from scipy.ndimage import center_of_mass
from pathlib import Path
import random
import pandas as pd
import subprocess
import shlex
import shutil

"""
Samples 3d dim_size x dim_size x dim_size patches per lesion (from center_of_mass) and saves NIfTIs
"""

ziel_ordner = "/path/to/folder/where/patches/should/be/stored/"

quell_ordner = "/path/where/images/are/saved/"

Path(ziel_ordner + "16/test/enhancing/").mkdir(parents=True, exist_ok=True)
Path(ziel_ordner + "16/test/non_enhancing/").mkdir(parents=True, exist_ok=True)
Path(ziel_ordner + "16/valid/enhancing/").mkdir(parents=True, exist_ok=True)
Path(ziel_ordner + "16/valid/non_enhancing/").mkdir(parents=True, exist_ok=True)
Path(ziel_ordner + "16/train/enhancing/").mkdir(parents=True, exist_ok=True)
Path(ziel_ordner + "16/train/non_enhancing/").mkdir(parents=True, exist_ok=True)

patienten = os.listdir(quell_ordner)
patienten = [pat for pat in patienten if os.path.isdir(quell_ordner + pat)]
patienten.sort()

res_df = pd.DataFrame(columns = ("ID","Save_Path","Enhancing","Label","Volume_mm3","t1sub_p50","t1sub_p75"))

for pat in patienten:
	exams = os.listdir(quell_ordner + pat + "/")
	exams.sort()

	if len(exams) > 3: #These are training / valid cases
		for exam in exams:

			if exam == exams[0]:
				f2_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-T1_FLAIR.nii.gz"
				t1_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-org_T1.nii.gz"
				t1c_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-T1_T1C.nii.gz" #Skullstripped
				seg_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-org_seg-lesionBLmanual_msk.nii.gz"
			else:
				f2_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-mni_FLAIR.nii.gz" #Skullstripped
				t1_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-org_T1.nii.gz"
				t1c_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-org_T1C.nii.gz"
				seg_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-nmi_seg-newlesionFUmanual_msk.nii.gz"

			if (os.path.exists(f2_file)) and (os.path.exists(t1_file)) and (os.path.exists(t1c_file)) and (os.path.exists(seg_file)):
				shutil.copy2(f2_file,"/home/bene/Documents/tmp/f2.nii.gz")
				shutil.copy2(t1_file,"/home/bene/Documents/tmp/t1.nii.gz")
				shutil.copy2(t1c_file,"/home/bene/Documents/tmp/t1c.nii.gz")
				shutil.copy2(seg_file,"/home/bene/Documents/tmp/seg.nii.gz")
				
				#Check if there is a lesion at all, otherwise skip (esp. registration!)
				seg_nib = nib.load("/home/bene/Documents/tmp/seg.nii.gz")
				seg = seg_nib.get_fdata()
				if seg.max() > 0:

					if exam != exams[0]: # We need to co-register stuff then
						reg_t1_call = "reg_aladin -ref /home/bene/Documents/tmp/f2.nii.gz -flo /home/bene/Documents/tmp/t1.nii.gz -rigOnly -pad 0 -res /home/bene/Documents/tmp/t1.nii.gz"
						subprocess.run(shlex.split(reg_t1_call),stdout=subprocess.PIPE,shell=False)

						reg_t1c_call = "reg_aladin -ref /home/bene/Documents/tmp/f2.nii.gz -flo /home/bene/Documents/tmp/t1c.nii.gz -rigOnly -pad 0 -res /home/bene/Documents/tmp/t1c.nii.gz"
						subprocess.run(shlex.split(reg_t1c_call),stdout=subprocess.PIPE,shell=False)

					f2_nib = nib.load("/home/bene/Documents/tmp/f2.nii.gz")
					t1_nib = nib.load("/home/bene/Documents/tmp/t1.nii.gz")
					t1c_nib = nib.load("/home/bene/Documents/tmp/t1c.nii.gz")

					f2 = f2_nib.get_fdata()
					f2 = np.nan_to_num(f2,nan=0.0, posinf=0.0, neginf=0.0)
					t1 = t1_nib.get_fdata()
					t1 = np.nan_to_num(t1,nan=0.0, posinf=0.0, neginf=0.0)
					t1c = t1c_nib.get_fdata()
					t1c = np.nan_to_num(t1c,nan=0.0, posinf=0.0, neginf=0.0)

					if exam == exams[0]:
						temp_bm = np.zeros(t1c.shape)
						temp_bm[t1c != t1c[1,1,1]] = 1
						f2[temp_bm == 0] = 0
						t1[temp_bm == 0] = 0
						t1c[temp_bm == 0] = 0
					else:
						temp_bm = np.zeros(f2.shape)
						temp_bm[f2 != 0] = 1
						t1[temp_bm == 0] = 0
						t1c[temp_bm == 0] = 0

					t1c = t1c - t1c[temp_bm == 1].min()
					t1c = np.clip(t1c,a_min=None,a_max=np.percentile(t1c[temp_bm == 1],99.9))
					t1c = np.divide(t1c,t1c[temp_bm == 1].max())
					t1c[temp_bm == 0] = 0

					t1 = t1 - t1[temp_bm == 1].min()
					t1 = np.clip(t1,a_min=None,a_max=np.percentile(t1[temp_bm == 1],99.9))
					t1 = np.divide(t1,t1[temp_bm == 1].max())
					t1[temp_bm == 0] = 0

					t1sub = t1c - t1

					f2 = f2 - f2[temp_bm == 1].min()
					f2 = np.clip(f2,a_min=None,a_max=np.percentile(f2[temp_bm == 1],99.9))
					f2 = np.divide(f2,f2[temp_bm == 1].max())
					f2[temp_bm == 0] = 0

					if seg.max() > 0:
						seg_label = label(seg,background=0, connectivity=3)
						seg_label = remove_small_objects(seg_label, min_size=7, connectivity=3)
						seg_label = label(seg_label,background=0, connectivity=3)

						for lesion in range(seg_label.max()):
							res_dict = {}
							lesion += 1 #We start with "1", not "0"!
							seg_temp = np.zeros(seg.shape)
							seg_temp[seg_label == lesion] = 1
							c_o_m = center_of_mass(seg_temp)
							sag = int(c_o_m[0])
							cor = int(c_o_m[1])
							ax = int(c_o_m[2])

							f2_patch_16 = f2[sag - (16 // 2):sag + (16 // 2),cor - (16 // 2):cor + (16 // 2),ax - (16 // 2):ax + (16 // 2)]
							t1_patch_16 = t1[sag - (16 // 2):sag + (16 // 2),cor - (16 // 2):cor + (16 // 2),ax - (16 // 2):ax + (16 // 2)]
							t1sub_patch_16 = t1sub[sag - (16 // 2):sag + (16 // 2),cor - (16 // 2):cor + (16 // 2),ax - (16 // 2):ax + (16 // 2)]

							if random.random() < .85:
								save_path = "train/"
							else:
								save_path = "valid/"

							if seg[seg_temp == 1].max() > 4:
								save_path = save_path + "enhancing/"
								res_dict["Enhancing"] = 1
							else:
								save_path = save_path + "non_enhancing/"
								res_dict["Enhancing"] = 0

							nib.save(nib.Nifti1Image(f2_patch_16.astype(np.float32),seg_nib.affine),ziel_ordner + "16/" + save_path + pat + "_" + exam + "_sequ-f2_lesion" + str(lesion) + ".nii.gz")
							nib.save(nib.Nifti1Image(t1_patch_16.astype(np.float32),seg_nib.affine),ziel_ordner + "16/" + save_path + pat + "_" + exam + "_sequ-t1_lesion" + str(lesion) + ".nii.gz")
							nib.save(nib.Nifti1Image(t1sub_patch_16.astype(np.float32),seg_nib.affine),ziel_ordner + "16/" + save_path + pat + "_" + exam + "_sequ-subt1_lesion" + str(lesion) + ".nii.gz")
							res_dict["ID"] = pat + "_" + exam + "_sequ-f2_lesion" + str(lesion) + ".nii.gz"
							res_dict["Save_Path"] = save_path
							res_dict["Label"] = seg[seg_temp == 1].max()
							res_dict["Volume_mm3"] = seg_temp.sum()
							res_dict["t1sub_p50"] = np.percentile(t1sub[seg_temp == 1],50)
							res_dict["t1sub_p75"] = np.percentile(t1sub[seg_temp == 1],75)

							res_df = res_df.append(res_dict,ignore_index=True)

				os.remove("/home/bene/Documents/tmp/f2.nii.gz")
				os.remove("/home/bene/Documents/tmp/t1.nii.gz")
				os.remove("/home/bene/Documents/tmp/t1c.nii.gz")
				os.remove("/home/bene/Documents/tmp/seg.nii.gz")

	else: #These are testing cases!
		exam = exams[0] # We only want baseline!
		f2_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-T1_FLAIR.nii.gz"
		t1_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-org_T1.nii.gz"
		t1c_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-T1_T1C.nii.gz" #Skullstripped
		seg_file = quell_ordner + pat + "/" + exam + "/" + pat + "_" + exam + "_space-org_seg-lesionBLmanual_msk.nii.gz"

		if (os.path.exists(f2_file)) and (os.path.exists(t1_file)) and (os.path.exists(t1c_file)) and (os.path.exists(seg_file)):
			shutil.copy2(f2_file,"/home/bene/Documents/tmp/f2.nii.gz")
			shutil.copy2(t1_file,"/home/bene/Documents/tmp/t1.nii.gz")
			shutil.copy2(t1c_file,"/home/bene/Documents/tmp/t1c.nii.gz")
			shutil.copy2(seg_file,"/home/bene/Documents/tmp/seg.nii.gz")

			f2_nib = nib.load("/home/bene/Documents/tmp/f2.nii.gz")
			t1_nib = nib.load("/home/bene/Documents/tmp/t1.nii.gz")
			seg_nib = nib.load("/home/bene/Documents/tmp/seg.nii.gz")
			t1c_nib = nib.load("/home/bene/Documents/tmp/t1c.nii.gz")

			f2 = f2_nib.get_fdata()
			f2 = np.nan_to_num(f2,nan=0.0, posinf=0.0, neginf=0.0)
			t1 = t1_nib.get_fdata()
			t1 = np.nan_to_num(t1,nan=0.0, posinf=0.0, neginf=0.0)
			t1c = t1c_nib.get_fdata()
			t1c = np.nan_to_num(t1c,nan=0.0, posinf=0.0, neginf=0.0)
			seg = seg_nib.get_fdata()

			temp_bm = np.zeros(t1c.shape)
			temp_bm[t1c != t1c[1,1,1]] = 1
			f2[temp_bm == 0] = 0
			t1[temp_bm == 0] = 0
			t1c[temp_bm == 0] = 0

			t1c = t1c - t1c[temp_bm == 1].min()
			t1c = np.clip(t1c,a_min=None,a_max=np.percentile(t1c[temp_bm == 1],99.9))
			t1c = np.divide(t1c,t1c[temp_bm == 1].max())
			t1c[temp_bm == 0] = 0

			t1 = t1 - t1[temp_bm == 1].min()
			t1 = np.clip(t1,a_min=None,a_max=np.percentile(t1[temp_bm == 1],99.9))
			t1 = np.divide(t1,t1[temp_bm == 1].max())
			t1[temp_bm == 0] = 0

			t1sub = t1c - t1

			f2 = f2 - f2[temp_bm == 1].min()
			f2 = np.clip(f2,a_min=None,a_max=np.percentile(f2[temp_bm == 1],99.9))
			f2 = np.divide(f2,f2[temp_bm == 1].max())
			f2[temp_bm == 0] = 0

			if seg.max() > 0:
				seg_label = label(seg,background=0, connectivity=3)
				seg_label = remove_small_objects(seg_label, min_size=7, connectivity=3)
				seg_label = label(seg_label,background=0, connectivity=3)

				for lesion in range(seg_label.max()):
					res_dict = {}
					lesion += 1 #We start with "1", not "0"!
					seg_temp = np.zeros(seg.shape)
					seg_temp[seg_label == lesion] = 1
					c_o_m = center_of_mass(seg_temp)
					sag = int(c_o_m[0])
					cor = int(c_o_m[1])
					ax = int(c_o_m[2])

					f2_patch_16 = f2[sag - (16 // 2):sag + (16 // 2),cor - (16 // 2):cor + (16 // 2),ax - (16 // 2):ax + (16 // 2)]
					t1_patch_16 = t1[sag - (16 // 2):sag + (16 // 2),cor - (16 // 2):cor + (16 // 2),ax - (16 // 2):ax + (16 // 2)]
					t1sub_patch_16 = t1sub[sag - (16 // 2):sag + (16 // 2),cor - (16 // 2):cor + (16 // 2),ax - (16 // 2):ax + (16 // 2)]

					save_path = "test/"

					if seg[seg_temp == 1].max() > 4:
						save_path = save_path + "enhancing/"
						res_dict["Enhancing"] = 1
					else:
						save_path = save_path + "non_enhancing/"
						res_dict["Enhancing"] = 0

					nib.save(nib.Nifti1Image(f2_patch_16.astype(np.float32),seg_nib.affine),ziel_ordner + "16/" + save_path + pat + "_" + exam + "_sequ-f2_lesion" + str(lesion) + ".nii.gz")
					nib.save(nib.Nifti1Image(t1_patch_16.astype(np.float32),seg_nib.affine),ziel_ordner + "16/" + save_path + pat + "_" + exam + "_sequ-t1_lesion" + str(lesion) + ".nii.gz")
					nib.save(nib.Nifti1Image(t1sub_patch_16.astype(np.float32),seg_nib.affine),ziel_ordner + "16/" + save_path + pat + "_" + exam + "_sequ-subt1_lesion" + str(lesion) + ".nii.gz")
					res_dict["ID"] = pat + "_" + exam + "_sequ-f2_lesion" + str(lesion) + ".nii.gz"
					res_dict["Save_Path"] = save_path
					res_dict["Label"] = seg[seg_temp == 1].max()
					res_dict["Volume_mm3"] = seg_temp.sum()
					res_dict["t1sub_p50"] = np.percentile(t1sub[seg_temp == 1],50)
					res_dict["t1sub_p75"] = np.percentile(t1sub[seg_temp == 1],75)

					res_df = res_df.append(res_dict,ignore_index=True)
			
			os.remove("/home/bene/Documents/tmp/f2.nii.gz")
			os.remove("/home/bene/Documents/tmp/t1.nii.gz")
			os.remove("/home/bene/Documents/tmp/t1c.nii.gz")
			os.remove("/home/bene/Documents/tmp/seg.nii.gz")

	print(pat + " done!")
res_df.to_csv(ziel_ordner + "lesion_patches.csv",index=False)