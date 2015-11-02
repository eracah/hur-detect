#Data Set#
###/global/project/projectdirs/nervana/yunjie/dataset###
Uses symbolic link to refer the data is probably a better way than copying data around.  

###localization task
-------------------------------------------------------------------
Data are in the "localization" sub folder

###>> Hurricanes
hurricanes_localization.h5 is contains hurricanes(~25000) and no hurricanes(~25000) training data.
Dimensions of images are 96x96. Diagnostic variables are TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT. For each of the positive example, there is hurricane in the image but not centered.


###classification task
-------------------------------------------------------------------

###>>Fronts
fronts_all.h5 contains fronts (5600) and no_fronts(28852) training data. Dimensions of images are 27x60. Diagnostic variables are Precipitation, pressure and temperature.

###>>Atmospheric River
####There are three subdirectories in this directory:old, new_landsea and new_merge   
The reason why I think about to calculate the TMQ and TMQ_land intersection and store it as a diagnostic variables is that this may save time for recalculating the TMQ_land intersection each time run the model. Trim the images to same size and merge the Europe and America data together potentially increase training data size, which can be beneficial on preventing overfitting and aiding model to learn. The method of calculating TMQ and TMQ_land intersection is adopted from Joaquin's python code.

###old: 
data that we were using for AGU abstract submission.  
1. atmosphericriver_us+TMQ_Jun8.h5 contains Atmospheric river (2304) and Non Atmospheric river (3077) training data for West coast of America. Dimensions of images are 158X224. The only diagnostic variable in the file is "Vertical Integrated Water Vapor", also called "TMQ".  
2. landmask_imgs_us.pkl contains the land sea mask image of the west cost of America. Image dimensions are 158x224. Value 1 indicates ocean, value 0 indicates land.  
3. atmosphericriver_eu+TMQ_Jun29.h5 contains Atmospheric river (4514) and Non Atmospheric river (3450) training data for west Europe. Dimensions of the images are 140x240. The only diagnostic variable in the file is "Vertical Integrated Water Vapor", also called "TMQ".  
4. landmask_imgs_eu.pkl contains the land sea mask image of the west Europe. Image dimensions are 140x240. Value 1 indicates ocean, Value 0 indicates land.  
###new_landsea: 
calculated the TMQ and land intersection and write it as additional diagnostic variable.  
1. atmosphericriver_us+TMQ+land_Sep4.h5 contains Atmospheric river (2304) and Non Atmospheric river (3077) training data for west coast of America. Dimensions of images are 158x224. Diagnostic variables are TMQ and TMQ_land intersection.  
2. atmosphericriver_eu+TMQ+land_Sep4.h5 contains Atmospheric river (4514) and Non Atmospheric river (3450) training data for west Europe. Dimensions of images are 140x240. Diagnostic variables are TMQ and TMQ_land intersection.  
###new_merge: 
trimmed the images of Atmospheric river, Non Atmospheric river, land sea mask of both the West coast of America and West Europe to be the same size and merged into one single file.  
1. atmospheric_river_us+eu+landsea_sep10.h5  contains Atmospheric river (6818) and Non Atmospheric river (6527) training data combined from west coast of America and west Europe. Dimensions of images are 148x224. Diagnostic variables are TMQ and TMQ_land interaction.  
      
###>>Hurricane  
hurricanes.h5 contains a bunch of hurricane (99925) and non hurricane (99899) training data. Dimensions of images are 32x32. Diagnostic variables are TMQ, V850, PSL, U850, T500, UBOT, T200, VBOT. 






Repo for climate classification using deep learning
Put raw data (h5 files and landmasks) in raw_data directory, but do not push to repo.