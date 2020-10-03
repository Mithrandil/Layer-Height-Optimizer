# Layer-Height-Optmizer
Smart Layer Height based on horizontal surfaces (.3mf files)

Usage:  LH-optimizer.py <file.3mf> <default_layer_height_in_mm> [max_layer_height_in_mm]

It's quite annoying to be forced to design objects with the flat surfaces height based on the layer height at which they will be printed (Specially now that the layer height table has been removed)

(e.g. slicing a 10,1 mm high cube with 0,20 mm layer height leads to a 10.2 mm high cube)

So, i suggest a smart modification of the layer height based on the distance of flat surfaces from the heated bed, exploiting the layer editing feature to make it automatic,

(for instance, slicing a 10,1 mm high cube at "smart" 0.20 mm layer height could lead to 41 layers 0.20 mm thick + 10 layers 0.19 mm thick --> 10.1 mm high cube)

New version of the script, with some improvements:

    Works directly with .3mf files instead of .stl (models can be oriented and scaled before optimization)

    Saves the layers directly in a new .3mf file named OPTLH_old.file_name.3mf

    Optional Parameter: max_layer_height (usually you want to set it at 0.75*nozzle-width)

