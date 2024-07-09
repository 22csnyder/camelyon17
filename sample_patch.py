

import multiresolutionimageinterface as mir
reader = mir.MultiResolutionImageReader()
#fname='/home/chris/Data/Camelyon/centre_1/patient_020/patient_020_node_4.tif'
fname='/media/chris/Untitled/CAMELYON_2017/centre_1/patient_020_node_4.tif'
mr_image = reader.open(fname) # 2ms


level = 2
ds = mr_image.getLevelDownsample(level)  #0.2 ms


#level, time:
# 0   , 6ms
# 1   , 3ms
# 2   , 1.5 ms
# 3   , 5ms

#shape: (200,300,3)

#Experiements show:
    #topleft row, topleft col, n_rows_output, n_col_output, downsamplelevel
image_patch = mr_image.getUCharPatch(int(568 * ds), int(732*ds),300,200,level)

