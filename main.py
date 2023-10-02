import os
import shutil
import datetime
from extract import extractFromAdani, extractFromTata, extractFromBSES, extractFromMSEB, extractFromReliance
# from classification import classificationFromImg
from classification_resnet import classificationFromImgByResnet

from post import post_processing
import json

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
        Check = classificationFromImgByResnet(img_path)
        if Check == 0:
            out = extractFromAdani(img_path)
            results[i] = out
            # adani_imgs.append(i)
            # shutil.copy(img_path, os.path.join(anadiDir, i))
            # print(out)
        elif Check == 1:
            out = extractFromBSES(img_path)
            results[i] = out
            # shutil.copy(img_path, os.path.join(bsesDir, i))
        elif Check == 2:
            out = extractFromMSEB(img_path)
            results[i] = out            
            # shutil.copy(img_path, os.path.join(msebDir, i))
        elif Check == 3:
            out = extractFromReliance(img_path)
            results[i] = out            
            # shutil.copy(img_path, os.path.join(reliDir, i))  
        else:
            out = extractFromTata(img_path)
            results[i] = out            
            # shutil.copy(img_path, os.path.join(tataDir, i))  
        
    with open(f'{output_dir}/result.json', 'w', encoding='utf-8') as fp:
        json.dump(results, fp, sort_keys=True, indent='\t', separators=(',', ': '))  
    post_processing(results, f'{output_dir}/result.xlsx')
            
    duration = datetime.datetime.now() - start
    # print(f"Adani: {len(adani_imgs)}, Not-Adani: {len(img_list)-len(adani_imgs)}")
    print(f"Time taken: {duration}")
    
    return results

if __name__ == "__main__":

    # Key paths and parameters
    DATA_DIR = "inputs"
    OUTPUT_DIR = "results"
    ERR_DIR = "Not-Adani"

#     # # Initialize logger
#     # if os.path.exists('parse_table.log'):
#     #     os.remove('parse_table.log')
#     # logger = logging.getLogger('parse_table')
#     # logger.setLevel(logging.INFO)
#     # ch = logging.StreamHandler()
#     # fh = logging.FileHandler('parse_table.log')
#     # logger.addHandler(ch)
#     # logger.addHandler(fh)

#     # Run main control flow    
    main(DATA_DIR, OUTPUT_DIR, ERR_DIR)

    
    
    

