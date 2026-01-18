"""
Created on Thu Apr  5 13:38:16 2018
Modified to support absolute paths and auto-detection.
"""

import os  # Adaugat pentru manipularea cailor

import imageio
import numpy as np


def make_epiinput(image_path, seq1, image_height, image_width, view_n, RGB):
    traindata_tmp = np.zeros(
        (1, image_height, image_width, len(view_n)), dtype=np.float32
    )
    i = 0
    # Gestionare input tip lista sau string
    if isinstance(image_path, list) and len(image_path) == 1:
        image_path = image_path[0]

    for seq in seq1:
        # Folosim os.path.join pentru compatibilitate (Linux/Windows)
        file_name = 'input_Cam0%.2d.png' % seq
        full_path = os.path.join(image_path, file_name)

        try:
            tmp = np.float32(imageio.imread(full_path))
            traindata_tmp[0, :, :, i] = (
                RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]
            ) / 255
            i += 1
        except FileNotFoundError:
            print(f'Eroare: Nu am gasit fisierul: {full_path}')
            # Putem alege sa continuam sau sa oprim executia. Aici continuam cu zerouri.
            i += 1

    return traindata_tmp


def make_epiinput_lytro(image_path, seq1, image_height, image_width, view_n, RGB):
    traindata_tmp = np.zeros(
        (1, image_height, image_width, len(view_n)), dtype=np.float32
    )

    i = 0
    if isinstance(image_path, list) and len(image_path) == 1:
        image_path = image_path[0]

    # Extragem numele folderului corect indiferent de calea absoluta
    folder_name = os.path.basename(os.path.normpath(image_path))

    for seq in seq1:
        # Constructie nume fisier specific Lytro
        file_name = '%s_%02d_%02d.png' % (
            folder_name,
            1 + seq // 9,
            1 + seq - (seq // 9) * 9,
        )
        full_path = os.path.join(image_path, file_name)

        try:
            tmp = np.float32(imageio.imread(full_path))
            traindata_tmp[0, :, :, i] = (
                RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]
            ) / 255
            i += 1
        except FileNotFoundError:
            print(f'Eroare Lytro: Nu am gasit fisierul: {full_path}')
            i += 1

    return traindata_tmp


def make_multiinput(image_path, image_height, image_width, view_n):
    RGB = [0.299, 0.587, 0.114]  ## RGB to Gray

    slice_for_5x5 = int(0.5 * (9 - len(view_n)))

    seq90d = list(range(4, 81, 9)[::-1][slice_for_5x5 : 9 - slice_for_5x5 :])
    seq0d = list(range(36, 45, 1)[slice_for_5x5 : 9 - slice_for_5x5 :])
    seq45d = list(range(8, 80, 8)[::-1][slice_for_5x5 : 9 - slice_for_5x5 :])
    seqM45d = list(range(0, 81, 10)[slice_for_5x5 : 9 - slice_for_5x5 :])

    # --- LOGICA NOUA PENTRU DETECTIE PATH ---

    # Normalizam calea (convertim lista in string daca e cazul)
    check_path = image_path
    if isinstance(check_path, list) and len(check_path) > 0:
        check_path = check_path[0]

    # 1. Verificam daca exista fisiere in format standard (HCI / Synthetic)
    # Cautam fisierul central (Cam040) ca indicator
    is_standard_hci = os.path.exists(os.path.join(check_path, 'input_Cam040.png'))

    # 2. Verificam daca e Lytro (bazat pe numele path-ului sau fallback)
    is_lytro = 'lytro' in check_path.lower()

    if is_standard_hci:
        # Modul Standard (folosit de obicei pentru demo-uri si synthetic)
        loader_func = make_epiinput
        # print(f"Format detectat: Standard HCI (in {check_path})")
    elif is_lytro:
        # Modul Lytro
        loader_func = make_epiinput_lytro
        # print(f"Format detectat: Lytro (in {check_path})")
    else:
        # Fallback: Incercam Standard daca nu suntem siguri
        print(
            'Atentie: Formatul imaginilor nu a fost detectat automat. Se incearca formatul Standard (input_CamXXX.png).'
        )
        loader_func = make_epiinput

    # Apelam functia selectata
    val_90d = loader_func(image_path, seq90d, image_height, image_width, view_n, RGB)
    val_0d = loader_func(image_path, seq0d, image_height, image_width, view_n, RGB)
    val_45d = loader_func(image_path, seq45d, image_height, image_width, view_n, RGB)
    val_M45d = loader_func(image_path, seqM45d, image_height, image_width, view_n, RGB)

    return val_90d, val_0d, val_45d, val_M45d
