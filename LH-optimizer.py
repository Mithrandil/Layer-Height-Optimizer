#!/usr/bin/python3

import numpy as np
from math import floor, sqrt
import sys
from lxml import etree as ET

import zipfile
import os
from os.path import basename, dirname

import tempfile

import argparse

# set to 2 to get 0.01 mm approximation
round_to = 2
# first_layer_height = 0.2
# min_layer_height = 0.05

top_layers = []


def setShrinkage(params):
    shrink = [0.0, 0.0]

    if params.shrinkage != 0:
        shrink[0] = params.shrinkage
        shrink[1] = params.shrinkage
    #elif params.auto_shrinkage_compensation:
        # calcola shrink in base al polimero, da fare
        # shrink = 0
    #    shrink[0] = 0.0
    #   shrink[1] = 0.0
    elif params.petg:
        shrink[0] = 0.198
        shrink[1] = 0.376
    elif params.asa:
        shrink[0] = 0.411
        shrink[1] = 0.411
    elif params.pla:
        shrink[0] = 0.103
        shrink[1] = 0.287
    return shrink


def calculate_layer_heights(layer_height, max_z):
    calculate_lh = True
    while calculate_lh:
        lh_list = LayerHeightList(round(max_z/100, round_to))
        for i in range(0, len(step_list) - 1):
            if args.debug:
                print("--------------------------------------------")
            max_z = step_list[i + 1]
            min_z = step_list[i]
            thickness = (max_z - min_z)

            # calculate ho many cents should be corrected

            layers = round(thickness / layer_height)
            if layers == 0:
                layers = 1

            if args.debug:
                print(i, " layers: ", layers)

            default_printed_thickness = layer_height * layers

            if args.debug:
                print(i, " thickness: ", thickness / 100)
                print(i, " default_printed_thickness: ", default_printed_thickness / 100)

            total_correction = thickness - default_printed_thickness

            if args.debug:
                print(i, " total_correction: ", total_correction / 100)

            avg_correction = total_correction / layers

            if args.debug:
                print(i, " avg_correction: ", avg_correction / 100)

            min_correction = floor(avg_correction)

            if args.debug:
                print(i, " min_correction: ", min_correction / 100)

            min_corrected_lh = layer_height + min_correction

            residual_correction = (total_correction - (min_correction * layers))

            if args.debug:
                print(i, " residual_correction: ", residual_correction / 100)

            layers_with_increased_correction = round(residual_correction / z_step)

            if args.debug:
                print(i, " layers_with_increased_correction: ", layers_with_increased_correction)

            layers_with_min_correction = layers - layers_with_increased_correction

            intermediate_z = min_z + (layers_with_min_correction * min_corrected_lh)

            if args.debug:
                print(i, " intermediate_z: ", intermediate_z / 100)

            if args.debug:
                print("------------------------------------------------------------> ", min_z / 100, " --> ",
                      intermediate_z / 100, " - ", min_corrected_lh / 100)

            # i.e. layers_with_min_correction != 0
            if min_z != intermediate_z:
                if args.debug:
                    print("------------------------------> ", "min_z != intermediate_z --- step_list[i]/100: ",
                          step_list[i] / 100)
                    print("------------------------------> ", "min_z/100: ", min_z / 100)
                    print("------------------------------> ", "min_corrected_lh/100: ", min_corrected_lh / 100)
                lh_list.add(min_z / 100, min_corrected_lh / 100)

            if residual_correction > 0:
                if args.debug:
                    print("------------------------------> ", "residual correction:", residual_correction)
                max_correction = min_correction + (residual_correction / abs(residual_correction))
                # print(i, " max_correction: ", max_correction)
                max_corrected_lh = layer_height + max_correction

                lh_list.add(intermediate_z / 100, max_corrected_lh / 100)

                if args.debug:
                    print("------------------------------------------------------------> ", intermediate_z / 100,
                          " --> ", max_z / 100, " - ", max_corrected_lh / 100)

        calculate_lh = False
        if max_layer_height:
            for layers_block in lh_list.get_list:
                if layers_block[2] > max_layer_height:
                    layer_height = layer_height - 1
                    print("Max Layer Height Exceeded,\nRecalculate with Default Layer Height ", layer_height / 100,
                          '\n')
                    calculate_lh = True
                    break
        if not calculate_lh:
            if len(lh_list.list_lh) > 1:
                print(str(layer_height / 100) + " - max-lh_diff:" + str(
                    round(max(lh_list.list_lh[1:]) - min(lh_list.list_lh[1:]), round_to)) + " - min:" + str(
                    min(lh_list.list_lh[1:])) + " max:" + str(max(lh_list.list_lh[1:])))
            return lh_list


# Function to find equation of plane.
# def calculate_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3):
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
    norm = sqrt(a * a + b * b + c * c)
    return a / norm, b / norm, c / norm


def add_dir_to_zip(zip_obj, tmp_dir_name, dir_name):
    for folderName, subfolders, filenames in os.walk(tmp_dir_name + '/' + dir_name):
        for filename in filenames:
            file_path = os.path.join(folderName, filename)
            zip_obj.write(file_path, file_path.replace(tmp_dir_name, ''))


class LayerHeightList:
    def __init__(self, max_z):
        self.list_z = []
        self.list_lh = []
        self.max_z = max_z
        self.previous_lh = 0.0
        self.single_perimeter_list = []

    def add(self, min_z, layer_height):
        min_z = round(min_z, round_to)
        layer_height = round(layer_height, round_to)
        if args.debug:
            print("LayerHeightList.add--> min_z: ", min_z, "layer_height: ", layer_height)
        if min_z in self.list_z:
            print("Error, this min_z was inserted already (", min_z, ")")
            quit()

        # gia verificato che serve anche oer single perimeter (altrimenti se la layer_height è uguale alla layer_height
        # precedente non inserisce lo step e poi non puo fare il single perimeter)
        # if len(self.list_lh) == 0 or layer_height != self.list_lh[-1] or args.single_perimeter or args.thin_top_layer:
        if len(self.list_lh) == 0 or layer_height != self.list_lh[-1] or args.single_perimeter or args.half_layer or args.thin_top_layer:
            self.list_z.append(min_z)
            self.list_lh.append(layer_height)
            self.previous_lh = layer_height

    @property
    def get_list(self):
        list_tmp = []
        z = self.list_z + [self.max_z]
        for i in range(0, len(self.list_lh)):
            list_tmp.append([z[i], z[i + 1], self.list_lh[i]])

            last_block_start = list_tmp[-1][0]
            last_block_end = list_tmp[-1][1]
            last_layer_height = list_tmp[-1][2]

            # add a last single layer if args.single_perimeter == True and there is enough thickness
            if args.single_perimeter or args.half_layer:
                if not args.thin_top_layer:
                    if round(last_block_end - last_block_start, round_to) >= round((2 * last_layer_height), round_to):
                        list_tmp[-1][1] = round(last_block_end - last_layer_height, round_to)
                        list_tmp.append([list_tmp[-1][1], last_block_end, last_layer_height])
                if round(last_block_end, round_to) in top_layers and round(last_block_end, round_to) > first_layer_height:
                    if args.half_layer and round(last_layer_height / 2, round_to) >= round(min_layer_height, round_to):
                        # new_layer_height = round(last_layer_height / 2, round_to)
                        # tolto arrotondamento perche dava problemi con la layer height dispari (e.g. 0.31 / 2 = 0.155)
                        new_layer_height = last_layer_height / 2
                        list_tmp[-1][2] = new_layer_height
                    if args.single_perimeter:
                        self.single_perimeter_list.append(round(last_block_end - last_layer_height, round_to))

        return list_tmp


if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0],
          " <file.3mf> [-o] [-s] [-hl] [-ttl] [--shrinkage <shrinkage>]")
    print("Example: ", sys.argv[0], " myfile.3mf -s")
    quit()

# Crea un oggetto parser
parser = argparse.ArgumentParser()

# Aggiungi un argomento di tipo flag
parser.add_argument('-s', '--single-perimeter', action='store_true', help='Single Perimeter Top')
parser.add_argument('-o', '--optimize', action='store_true', help='Optimize Layer Height')
parser.add_argument('-hl', '--half-layer', action='store_true', help='Half LayerHeight Top')
parser.add_argument('-ttl', '--thin-top-layer', action='store_true', help='Thin Top Layer')
parser.add_argument('-d', '--debug', action='store_true', help='Debug')
parser.add_argument('-petg', '--petg', action='store_true', help='PETG (for shrinkage)')
parser.add_argument('-pla', '--pla', action='store_true', help='PLA (for shrinkage)')
parser.add_argument('-asa', '--asa', action='store_true', help='ASA (for shrinkage)')
# parser.add_argument('-asc', '--auto-shrinkage-compensation', action='store_true', help='Auto-shrinkage-compensation')

# Aggiungi argomenti di tipo float
parser.add_argument('--min', type=float, default=0.0, help='Valore minimo')
parser.add_argument('--max', type=float, default=0.0, help='Valore massimo')
parser.add_argument('--avg', type=float, default=0.0, help='Valore medio')
parser.add_argument('--shrinkage', type=float, default=0.0, help='Shrinkage %')

# Aggiungi l'argomento per il nome del file
parser.add_argument('file', help='Nome del file')

# Parsa gli argomenti della riga di comando
args = parser.parse_args()

# Valuta gli argomenti float
min_layer_height = args.min
max_layer_height = args.max
avg_layer_height = args.avg

shrinkage = setShrinkage(args)

# Ottieni il nome del file
file_name = args.file

with tempfile.TemporaryDirectory() as tmpdirname:
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(tmpdirname)

    # Carica il file di configurazione
    with open(tmpdirname + '/Metadata/Slic3r_PE.config', 'r') as config_file:
        lines = config_file.readlines()

        for line in lines:
            if min_layer_height == 0.0 and line.startswith("; min_layer_height"):
                min_layer_height = list(map(float, line.split(" = ")[1].split(",")))[0]
            elif line.startswith("; first_layer_height"):
                first_layer_height = list(map(float, line.split(" = ")[1].split(",")))[0]
            elif avg_layer_height == 0.0 and line.startswith("; layer_height"):
                avg_layer_height = list(map(float, line.split(" = ")[1].split(",")))[0]
            elif max_layer_height == 0.0 and line.startswith("; max_layer_height"):
                # max_layer_height = float(line.split(" = ")[1])
                max_layer_height = list(map(float, line.split(" = ")[1].split(",")))[0]

        min_layer_height_cents = min_layer_height * 100

    if avg_layer_height < 0.01 or avg_layer_height > 2:
        print("Error, layer_height out of limits")
        print("Usage: ", sys.argv[0],
              " [-s] [-hl] [-o] [-d] <file.3mf> [--avg <default_layer_height_in_mm>] [--min <min_layer_height_in_mm>] [--max <max_layer_height_in_mm>]")
        print("Example: ", sys.argv[0], " -s myfile.3mf --avg 0.20 --max 0.25")
        quit()

    # better not to change z_step_mm it,
    # some other approximations rely on this value to be set to 0.01 mm
    # changing it would require rewrite of other code
    # notice that Prusa Slicer does not support values with precision higher than 0.01 mm
    z_step_mm = 0.01

    first_layer_height_cents = first_layer_height * 100
    z_step = z_step_mm * 100

    print("--------------------------------------------------------")
    print("Optimal Layer Height calculator with 0.01 mm approximation, parameters used:")
    print("First Layer Height: ", first_layer_height, " mm")
    print("Default Layer Height: ", avg_layer_height, " mm")
    print("Min Layer Height: ", min_layer_height, " mm")
    print("Max Layer Height: ", max_layer_height, " mm")
    print("Z-Steps and Height Approximation: ", z_step_mm, " mm")
    print("--------------------------------------------------------")

    # -------------NEW-----------------------------------------

    treeConfig = ET.parse(tmpdirname + '/Metadata/Slic3r_PE_model.config')
    rootConfig = treeConfig.getroot()

    # Itera sugli oggetti e estrai layer_height
    '''
    for obj in rootConfig.iter('object_element'):
        layer_height_elem = obj.find("./metadata[@key='layer_height']")
        if layer_height_elem is not None:
            layer_height = layer_height_elem.get('value')
            print(f"Layer Height: {layer_height}")
        else:
            print("Layer Height non disponibile per questo oggetto")
    '''
    # -----------------------------------------------------------------------------------------------
    # create the XML file structure for the .3mf file
    # not really needed, but useful for complicated meshes
    # unfortunately it's not simple to create the whole .3mf file starting from scratch, so it's better to just put this file into the Metadata folder inside the .3mf file saved from PrusaSlicer (which is actually a zip file)

    objects = ET.Element('objects')
    outtree = ET.ElementTree(objects)

    tree = ET.parse(tmpdirname + '/3D/3dmodel.model')
    root = tree.getroot()

    transforms = []

    for elem in root:
        for model in elem:
            if 'transform' in model.attrib:
                transform = np.matrix(model.attrib['transform']).copy()
                transform.resize(4, 3)
                transform = np.c_[transform, [0, 0, 0, 1]]
                transforms.append(np.array(transform.transpose()))

    expansion_matrix = np.array([
        [1.0 + shrinkage[0] / 100, 0, 0, 0],
        [0, 1.0 + shrinkage[1] / 100, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Get all 'item' elements
    items = root.findall('.//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}item')

    # Loop over all 'item' elements and update 'transform' attribute
    for item in items:
        transform_str = item.attrib['transform']
        transform_values = list(map(float, transform_str.split()))

        # Creiamo una matrice 4x4 con numpy. Iniziamo con un array di zeri e riempiamo le prime tre righe con i nostri valori
        transform_matrix = np.zeros((4, 4))
        transform_matrix[0:3, :] = np.array(transform_values).reshape(4, 3).T

        # Aggiungiamo l'ultima riga [0, 0, 0, 1]
        transform_matrix[3, 3] = 1

        # Calcolo la trasformazione e rimuovo l'ultima riga
        # transform = np.dot(transform_matrix, expansion_matrix)[:-1]
        transform = np.dot(expansion_matrix, transform_matrix)[:-1]

        # Update the 'transform' attribute
        item.attrib['transform'] = ' '.join(map(str, transform.T.flatten().tolist()))

    # Salva il file XML modificato
    tree.write(tmpdirname + '/3D/3dmodel.model', encoding='utf-8', xml_declaration=True)

    for elem in root:
        for model in elem:
            if 'type' in model.attrib and model.attrib['type'] == 'model':
                mesh = model.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}mesh')
                if mesh is None:
                    continue
                objid = model.attrib['id']

                print(" ")
                print("OBJECT-", objid, '--------------------------------------------------\n')

                lh = avg_layer_height * 100
                # Itera sugli oggetti e estrai layer_height
                # obj = rootConfig.find(f"./object_element[@id='{objid}']")
                obj = rootConfig.find(f"./object[@id='{objid}']")
                if obj is not None:
                    layer_height_elem = obj.find("./metadata[@key='layer_height']")
                    if layer_height_elem is not None:
                        lh = 100 * float(layer_height_elem.get('value'))
                        objname_elem = obj.find("./metadata[@key='name']")
                        if objname_elem is not None:
                            objname = objname_elem.get('value')
                        else:
                            objname = objid
                        print(f"\"{objname}\" Default Layer Height: {lh / 100} mm\n")
                else:
                    print("NO obj")

                vertices = mesh.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}vertices')

                set_z = set()

                numvert = []
                for vertice in vertices:
                    tempvert = np.dot(transforms[int(objid) - 1], np.array(
                        [float(vertice.attrib['x']), float(vertice.attrib['y']), float(vertice.attrib['z']), 1]))
                    numvert.append(tempvert)
                    set_z.add(tempvert[2])

                absolute_max_z = max(set_z)
                absolute_min_z = min(set_z)

                print("z range= ", round(absolute_min_z, round_to), 'mm - ', round(absolute_max_z, round_to), "mm\n")

                step_set = {0.0, first_layer_height_cents, round(absolute_max_z * 100)}

                triangles = mesh.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}triangles')
                for triangle in triangles:

                    vertexes = [numvert[int(triangle.attrib['v1'])], numvert[int(triangle.attrib['v2'])],
                                numvert[int(triangle.attrib['v3'])]]

                    normal = calculate_normal(vertexes)

                    if abs(normal[0]) < 0.0000001 and abs(normal[1]) < 0.0000001 and normal[2] != 0:

                        if round(vertexes[0][2], round_to) == round(vertexes[1][2], round_to) and round(
                                vertexes[1][2], round_to) == round(vertexes[2][2], round_to):

                            # trascura i layer più bassi del primo strato perchè non si vuole ridurre il first_layer_height sotto al valore impostato
                            if int(round((vertexes[0][2]) * 100)) >= first_layer_height_cents:
                                step_set.add(round(vertexes[0][2] * 100))

                        else:
                            print("ERROR ---------------------")
                            quit()

                step_list = sorted(step_set)
                print("Steps: ", [x / 100 for x in step_list], "\n")

                step_list_length = len(step_list)
                for step_index in range(1, step_list_length):
                    top_layers.append(round(step_list[step_index] / 100, round_to))

                if args.thin_top_layer:
                    for step_index in range(1, step_list_length):
                        if step_list[step_index] < first_layer_height_cents + min_layer_height_cents:
                            continue
                        if (step_list[step_index] - step_list[step_index - 1]) >= (2 * min_layer_height_cents):
                            step_list.append(step_list[step_index] - min_layer_height_cents)
                    step_list = sorted(step_list)

                if args.debug:
                    if args.thin_top_layer:
                        print("Thin top_layers: ", top_layers)
                    else:
                        print("top_layers: ", top_layers)
                    print(step_list)

                # insert a cycle which iterates over layer_height from min_layer_height to max_layer_height in steps of z_step_mm and prints the difference between the maximum and the minimum value in layer_steps_list.list_lh
                layer_steps_list = []
                min_layer_difference = max_layer_height - min_layer_height
                max_optimum_avg_layer_height = 0
                max_optimum_avg_layer_height_index = 0

                if args.avg > 0.0 or not args.optimize:
                    layer_steps_list.append(calculate_layer_heights(lh, absolute_max_z))
                    if len(layer_steps_list[-1].list_lh[1:]) > 0:
                        min_layer_difference = round(
                            max(layer_steps_list[-1].list_lh[1:]) - min(layer_steps_list[-1].list_lh[1:]), round_to)
                    else:
                        min_layer_difference = 0
                    max_optimum_avg_layer_height_index = len(layer_steps_list) - 1
                    max_optimum_avg_layer_height = avg_layer_height * 100

                    print("Fixed Average Layer Height: " + str(max_optimum_avg_layer_height / 100))
                else:
                    for lh in range(int(min_layer_height * 100), int(max_layer_height * 100), int(z_step_mm * 100)):

                        layer_steps_list.append(calculate_layer_heights(lh, absolute_max_z))
                        if len(layer_steps_list[-1].list_lh) <= 1:
                            min_layer_difference = 0
                            max_optimum_avg_layer_height_index = len(layer_steps_list) - 1
                            max_optimum_avg_layer_height = lh
                        elif round(max(layer_steps_list[-1].list_lh[1:]) - min(layer_steps_list[-1].list_lh[1:]),
                                   round_to) <= min_layer_difference:
                            min_layer_difference = round(
                                max(layer_steps_list[-1].list_lh[1:]) - min(layer_steps_list[-1].list_lh[1:]), round_to)
                            max_optimum_avg_layer_height_index = len(layer_steps_list) - 1
                            max_optimum_avg_layer_height = lh

                    print("Max Optimal Layer Height: " + str(max_optimum_avg_layer_height / 100))

                # print the list
                for line in layer_steps_list[max_optimum_avg_layer_height_index].get_list:
                    print(
                        "X-----------------------------------------------------------> " if args.debug else "",
                        line[0], " --> ", line[1], " - ", line[2],
                        "   <------   WARNING! Layer Height < " + str(min_layer_height) + " mm" if line[
                                                                                                       2] < min_layer_height else "",
                        "   <------   Layer Height > " + str(avg_layer_height) + " mm" if line[2] > float(
                            avg_layer_height) else "", )

                object_element = ET.SubElement(objects, 'object_element')
                object_element.set('id', objid)

                # <option opt_key="layer_height">0.2</option>
                # <option opt_key="perimeters">1</option>

                for line in layer_steps_list[max_optimum_avg_layer_height_index].get_list:

                    rng = ET.SubElement(object_element, 'range')
                    rng.set('min_z', str(line[0]))
                    rng.set('max_z', str(line[1]))

                    option1 = ET.SubElement(rng, 'option')
                    option1.set('opt_key', 'extruder')
                    option1.text = '0'

                    option2 = ET.SubElement(rng, 'option')
                    option2.set('opt_key', 'layer_height')
                    option2.text = str(line[2])

                    if line[0] in layer_steps_list[max_optimum_avg_layer_height_index].single_perimeter_list:
                        option3 = ET.SubElement(rng, 'option')
                        option3.set('opt_key', 'perimeters')
                        option3.text = '1'

    outtree.write(tmpdirname + "/Metadata/Prusa_Slicer_layer_config_ranges.xml", xml_declaration=True, encoding='utf-8')

    # create a ZipFile object_element
    with zipfile.ZipFile(os.path.join(dirname(file_name), 'OPTLH_' + basename(file_name)), 'w',
                         compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipObj:
        zipObj.write(tmpdirname + '/[Content_Types].xml', '[Content_Types].xml')
        add_dir_to_zip(zipObj, tmpdirname, '_rels')
        add_dir_to_zip(zipObj, tmpdirname, '3D')
        add_dir_to_zip(zipObj, tmpdirname, 'Metadata')

    print('\n\nCreated File: "', os.path.join(dirname(file_name), 'OPTLH_' + basename(file_name)), '"')
