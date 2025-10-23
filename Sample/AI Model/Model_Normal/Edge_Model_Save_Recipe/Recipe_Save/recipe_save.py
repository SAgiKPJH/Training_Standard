import os
import cv2
import traceback
import shutil

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


path = 'E:/Mirero/LHJ/Source Code/Edge_Model_Save_Recipe/Inline_Recipe/Dupont'
steps = os.listdir(path)

for step in steps :
    #if step == "COF":
    #    continue
    if step == 'dupont_unpattern_250526.py' :
        createFolder(f'{path}/{step[:-3]}/1/')
        shutil.move(f'{path}/{step}', f'{path}/{step[:-3]}/1/{step}')
