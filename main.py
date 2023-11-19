import os
import shutil
import datetime
from extract import extractFromAdani, extractFromTata, extractFromBSES, extractFromMSEB, extractFromReliance, extractFromBest
# from classification import classificationFromImg
from classification_resnet import classificationFromImgByResnet
from yolo_usage.assist import getTotalValue
from post import post_processing
import json
import cv2
def clear_contents(dir_path):
    '''
    Deletes the contents of the given filepath. Useful for testing runs.
    '''
    filelist = os.listdir(dir_path)
    if filelist:

        for f in filelist:
            if os.path.isdir(os.path.join(dir_path, f)):
                shutil.rmtree(os.path.join(dir_path, f))
            else:
                os.remove(os.path.join(dir_path, f))
    return None

def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass
def apiMain(img_path):
    print(f"Parsing file: {img_path}")
    weights = 'weights/epoch108.pt'
    # halfImg = img_path[0:-4] + '_1.jpg'
    # img = cv2.imread(img_path)
    # cv2.imwrite(img, img[0:int(img.shape[0]/2)])
    Check, xc, yc = getTotalValue(weights=weights, conf_thres=0.3, source=img_path)[0]
    # os.remove(halfImg)

    if Check is None:       
        Check = classificationFromImgByResnet(img_path)
    try:
        if Check == 0:
            out = extractFromAdani(img_path, xc, yc)
        elif Check == 1:
            out = extractFromBest(img_path, xc, yc)
        elif Check == 2:
            out = extractFromBSES(img_path, xc, yc)
        elif Check == 3:
            out = extractFromMSEB(img_path, xc, yc)
        elif Check == 4:
            out = extractFromReliance(img_path, xc, yc)            
        else:
            out = extractFromTata(img_path, xc, yc)
    except: 
        out = {}
    out['File_name'] = img_path.split("\\")[-1]
    return out  
def main(data_dir, output_dir, err_dir):
    '''
    Main control flow:
        1. Checks if required folders exist; if not, creates them
        2. Loops over each PDF file in data_path and calls parse_doc().
        3. Output xlsx files are written to output_path.
    '''
    classes = ['adani', 'bses', 'mseb', 'reliance', 'tata']
    anadiDir, bsesDir, tataDir, msebDir, reliDir = "results/ANADI", "results/BSES", "results/TATA", "results/MSEB", "results/RELIANCE"
    # Check if organizing folders exist
    for i in [data_dir, output_dir]:
        try:
            if i == data_dir and not os.path.exists(data_dir):
                raise Exception("Data folder is missing or not assigned.")
            else:
                os.mkdir(i)
        except FileExistsError:
            continue
    # Clear output folder
    clear_contents(output_dir)
    # clear_contents(err_dir)

    # makedir(anadiDir)
    # makedir(bsesDir)
    # makedir(tataDir)
    # makedir(msebDir)
    # makedir(reliDir)

    # clear_contents(anadiDir)
    # clear_contents(bsesDir)
    # clear_contents(msebDir)
    # clear_contents(reliDir)
    # clear_contents(tataDir)

    # Get list of pdfs to parse
    img_list = [f for f in os.listdir(data_dir) if (f.split('.')[-1].lower() in ['jpg', 'png', 'tiff', 'tif'])]
    img_list.sort()
    print(f"{len(img_list)} file(s) detected.")
    start = datetime.datetime.now()
    # Loop over PDF files, create Document objects, call Document.parse()
    cnt = 0
    results = {}
    for i in img_list:
        # result = {}
        cnt = cnt + 1
        img_path = os.path.join(data_dir, i)
        print(f"Parsing file_{cnt}/{len(img_list)}: {img_path}")

        weights = 'weights/epoch140.pt'
        halfImg = img_path[0:-4] + '_1.jpg'
        img = cv2.imread(img_path)
        cv2.imwrite(halfImg, img[0:int(img.shape[0]/2)])
        Check, xc, yc = getTotalValue(weights=weights, conf_thres=0.3, source=halfImg)[0]
        os.remove(halfImg)

        if Check is None:       
            Check = classificationFromImgByResnet(img_path)
        print(Check)
        if Check == 0:
            out = extractFromAdani(img_path, xc, yc)
            results[i] = out
            # adani_imgs.append(i)
            # shutil.copy(img_path, os.path.join(anadiDir, i))
            # print(out)
        elif Check == 1:
            out = extractFromBSES(img_path, xc, yc)
            results[i] = out
            # shutil.copy(img_path, os.path.join(bsesDir, i))
        elif Check == 2:
            out = extractFromMSEB(img_path, xc, yc)
            results[i] = out            
            # shutil.copy(img_path, os.path.join(msebDir, i))
        elif Check == 3:
            out = extractFromReliance(img_path, xc, yc)
            results[i] = out            
            # shutil.copy(img_path, os.path.join(reliDir, i))  
        else:
            out = extractFromTata(img_path, xc, yc)
            results[i] = out            
            # shutil.copy(img_path, os.path.join(tataDir, i))  
        
    with open(f'{output_dir}/result.json', 'w', encoding='utf-8') as fp:
        json.dump(results, fp, sort_keys=True, indent='\t', separators=(',', ': '))  
    post_processing(results, f'{output_dir}/result.xlsx')
            
    duration = datetime.datetime.now() - start
    # print(f"Adani: {len(adani_imgs)}, Not-Adani: {len(img_list)-len(adani_imgs)}")
    print(f"Time taken: {duration}")
    
    return results

# if __name__ == "__main__":

#     # Key paths and parameters
#     DATA_DIR = "inputs"
#     OUTPUT_DIR = "results"
#     ERR_DIR = "Failed"

#     main(DATA_DIR, OUTPUT_DIR, ERR_DIR)

    
    
    

