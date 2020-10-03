#!/usr/bin/python3

import numpy
from stl import mesh # from package numpy-stl
from math import floor, sqrt
import sys
import xml.etree.ElementTree as ET

import zipfile
import os
from os.path import basename, dirname

import tempfile

debug = False

#set to 2 to get 0.01 mm approximation
round_to = 2
first_lh_mm = 0.2
min_lh = 0.05

# Function to find equation of plane. 
#def calculate_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3):  
def calculate_normal(vects): 
   
    a1 = vects[1][0] - vects[0][0] 
    b1 = vects[1][1] - vects[0][1] 
    c1 = vects[1][2] - vects[0][2] 
    a2 = vects[2][0] - vects[0][0] 
    b2 = vects[2][1] - vects[0][1] 
    c2 = vects[2][2] - vects[0][2] 
    
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    norm = sqrt(a*a+b*b+c*c)
    return (a/norm, b/norm, c/norm)

def addDirToZip(zipObj, tmpdirname, dirname):
    for folderName, subfolders, filenames in os.walk(tmpdirname+'/'+dirname):
        for filename in filenames:
            filePath = os.path.join(folderName, filename)
            zipObj.write(filePath, filePath.replace(tmpdirname, ''))

class lh_List:
  def __init__(self, max_z):
    self.list_z = []
    self.list_lh = []
    self.max_z = max_z
    
  def add(self, min_z, lh):
    min_z = round(min_z, 6)
    lh = round(lh, 6)
    if min_z in self.list_z:
      print("Error, this min_z was inserted already (", min_z , ")")
      quit()
    if len(self.list_lh) == 0 or lh != self.list_lh[-1]:
      self.list_z.append(min_z)
      self.list_lh.append(lh)
      
  def getList(self):
    list_tmp = []
    z = self.list_z + [self.max_z]
    for i in range(0, len(self.list_lh)):
      list_tmp.append([z[i], z[i+1], self.list_lh[i]])
    return list_tmp

if len(sys.argv) < 3:
  print("Usage: ", sys.argv[0], " <file.3mf> <default_layer_height_in_mm> [max_layer_height_in_mm]")
  print("Example: ", sys.argv[0], " myfile.3mf 0.20 0.25")
  quit()
  
lh_mm = float(sys.argv[2])

if len(sys.argv) == 4:
    max_lh = float(sys.argv[3])
else:
    max_lh = None

with tempfile.TemporaryDirectory() as tmpdirname:
    with zipfile.ZipFile(sys.argv[1], 'r') as zip_ref:
        zip_ref.extractall(tmpdirname)


    if lh_mm < 0.01 or lh_mm > 2:
        print("Error, layer_height out of limits")
        print("Usage: ", sys.argv[0], " <file.3mf> <default_layer_height_in_mm> [max_layer_height_in_mm]")
        print("Example: ", sys.argv[0], " myfile.3mf 0.20 0.25")
        quit()

    # better not to change z_step_mm it, 
    # some other approximations rely on this value to be set to 0.01 mm
    # changing it would require rewrite of other code
    z_step_mm = 0.01

    lh = lh_mm * 100
    first_lh = first_lh_mm * 100
    z_step = z_step_mm * 100

    print("--------------------------------------------------------")
    print("Optimal Layer Height calculator with 0.01 mm approximation, parameters used:")
    print("First Layer Height: ", first_lh_mm, " mm", " (will not be changed)");
    print("Default Layer Height: ", lh_mm, " mm");
    print("Z-Steps and Height Approximation: ", z_step_mm, " mm");
    print("--------------------------------------------------------")

    #-------------NEW-----------------------------------------

    # -----------------------------------------------------------------------------------------------
    # create the XML file structure for the .3mf file
    # not really needed, but useful for complicated meshes
    # unfortunately it's not simple to create the whole .3mf file starting from scratch so it's better to just put this file into the Metadata folder inside the .3mf file saved from PrusaSlicer (which is actually a zip file)

    objects = ET.Element('objects')
    outtree = ET.ElementTree(objects)

    tree = ET.parse(tmpdirname+'/3D/3dmodel.model')
    root = tree.getroot()

    transforms = []

    for elem in root:
        for model in elem:
            if 'transform' in model.attrib:
                transform = numpy.matrix(model.attrib ['transform']).copy()
                transform.resize(4,3)
                transform = numpy.c_[transform, [0,0,0,1]]
                transforms.append(numpy.array(transform.transpose()))
                #print(transform)
                
                
    for elem in root:
        for model in elem:
            if 'type' in model.attrib and model.attrib['type'] == 'model':
                mesh = model.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}mesh')
                if mesh == None:
                    continue
                objid = model.attrib['id']
                print(" ")
                print("OBJECT-", objid, '--------------------------------------------------\n')
                vertices = mesh.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}vertices')

                set_z = set()
                    
                numvert = []
                for vertice in vertices:
                    tempvert = numpy.dot(transforms[int(objid)-1], numpy.array([float(vertice.attrib['x']), float(vertice.attrib['y']), float(vertice.attrib['z']), 1]))
                    numvert.append(tempvert)
                    set_z.add(tempvert[2])
                    
                absolute_max_z = max(set_z)
                absolute_min_z = min(set_z)
                
                print("z range= ", round(absolute_min_z, 6), 'mm - ', round(absolute_max_z, 6), "mm\n")

                step_set = set([0.0, first_lh, round((absolute_max_z)*100)])
                
                triangles = mesh.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}triangles')
                for triangle in triangles:
                    
                    vertexes = [numvert[int(triangle.attrib['v1'])], numvert[int(triangle.attrib['v2'])], numvert[int(triangle.attrib['v3'])]]
                    
                    normal = calculate_normal(vertexes)
                    
                    if abs(normal[0]) < 0.0000001 and abs(normal[1]) < 0.0000001 and normal[2] != 0:
                            
                            if round(vertexes[0][2],round_to) == round(vertexes[1][2], round_to) and round(vertexes[1][2], round_to) == round(vertexes[2][2], round_to):
                            
                                if int(round((vertexes[0][2])*100)) >= first_lh:
                                    step_set.add(round((vertexes[0][2])*100))
                            
                            else:
                                print("ERROR ---------------------")
                                quit()

                step_list = sorted(step_set)
                print("Steps: ", [x / 100 for x in step_list] , "\n")

                if debug == True:
                    print(step_list)

                

                calculate_lh = True
                while calculate_lh:
                    ls_list = lh_List(round(absolute_max_z, round_to))
                    for i in range(0, len(step_list)-1):
                        if debug == True:
                            print("--------------------------------------------")
                        max_z = step_list[i+1]
                        min_z = step_list[i]
                        thickness = (max_z - min_z)
                        
                        #calculate ho many cents should be corrected
                        
                        layers = round(thickness / lh)
                        if layers == 0:
                            layers = 1
                        
                        if debug == True:
                            print(i, " layers: ", layers)
                        
                        default_printed_thickness = lh * layers
                        
                        if debug == True:
                            print(i, " thickness: ", thickness / 100)
                            print(i, " default_printed_thickness: ", default_printed_thickness/100)
                        
                        total_correction = thickness - default_printed_thickness 

                        
                        if debug == True:
                            print(i, " total_correction: ", total_correction/100)
                        
                        avg_correction = total_correction / layers
                        
                        if debug == True:
                            print(i, " avg_correction: ", avg_correction/100)
                        
                        min_correction = floor(avg_correction)
                        
                        if debug == True:
                            print(i, " min_correction: ", min_correction/100)
                        
                        min_corrected_lh = lh + min_correction
                        
                        residual_correction = (total_correction - (min_correction*layers))
                        
                        if debug == True:
                            print(i, " residual_correction: ", residual_correction/100)
                        
                        layers_with_increased_correction = round(residual_correction / z_step)
                        
                        if debug == True:
                            print(i, " layers_with_increased_correction: ", layers_with_increased_correction)
                        
                        layers_with_min_correction = layers - layers_with_increased_correction
                        
                        intermediate_z = min_z + (layers_with_min_correction * min_corrected_lh)
                        
                        if debug == True:
                            print(i, " intermediate_z: ", intermediate_z/100)
                            
                        if debug == True:
                            print("------------------------------------------------------------> ", min_z/100, " --> ", intermediate_z/100 , " - ", min_corrected_lh/100)
                        
                        #i.e. layers_with_min_correction != 0
                        if min_z != intermediate_z:
                            ls_list.add(min_z/100, min_corrected_lh/100)
                        
                        if residual_correction > 0:
                            max_correction = min_correction + (residual_correction / abs(residual_correction))
                            #print(i, " max_correction: ", max_correction)
                            max_corrected_lh = lh + max_correction
                            
                            ls_list.add(intermediate_z/100, max_corrected_lh/100)
                            
                            if debug == True:
                                print("------------------------------------------------------------> ", intermediate_z/100, " --> ", max_z/100 , " - ", max_corrected_lh/100)
                            
                    calculate_lh = False  
                    if max_lh:
                        for line in ls_list.getList():
                            if line[2] > max_lh:
                                lh = lh - 1
                                print("Max Layer Height Exceeded,\nRecalculate with Default Layer Height ", lh/100, '\n')
                                calculate_lh = True
                                break
                    
                #print the list
                for line in ls_list.getList():
                    print("------------------------------------------------------------> " if debug == True else "", line[0], " --> ", line[1], " - ", line[2],
                    "   <------   WARNING! Layer Height < "+str(min_lh)+" mm" if line[2] < min_lh else "",
                    "   <------   Layer Height > "+str(lh_mm)+" mm" if line[2] > lh_mm else "",)

                object = ET.SubElement(objects, 'object')
                object.set('id',objid)

                for line in ls_list.getList():

                    rng = ET.SubElement(object, 'range')
                    rng.set('min_z', str(line[0]))
                    rng.set('max_z', str(line[1]))
                    
                    option1 = ET.SubElement(rng, 'option')
                    option1.set('opt_key','extruder')
                    option1.text = '0'

                    option2= ET.SubElement(rng, 'option')
                    option2.set('opt_key','layer_height')
                    option2.text = str(line[2])

    outtree.write(tmpdirname+"/Metadata/Prusa_Slicer_layer_config_ranges.xml", xml_declaration=True, encoding='utf-8')

    # create a ZipFile object
    with zipfile.ZipFile(os.path.join(dirname(sys.argv[1]),'OPTLH_'+basename(sys.argv[1])), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipObj:
        zipObj.write(tmpdirname+'/[Content_Types].xml', '[Content_Types].xml')
        addDirToZip(zipObj, tmpdirname, '_rels')
        addDirToZip(zipObj, tmpdirname, '3D')
        addDirToZip(zipObj, tmpdirname, 'Metadata')
    
    print('\n\nCreated File: "', os.path.join(dirname(sys.argv[1]),'OPTLH_'+basename(sys.argv[1])), '"')
