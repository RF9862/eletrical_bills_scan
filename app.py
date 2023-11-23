import os
import uvicorn
from fastapi.responses import FileResponse
from fastapi import (
    FastAPI,
    Request,
    Depends,
    Response,
    UploadFile,
    File, Form, HTTPException
)
from main import main, apiMain
import json
from fastapi.staticfiles import StaticFiles
import aiofiles
from typing import List
import shutil
import time
def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass    

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")
output = {}
filenames = []


@app.get("/")
async def read_index():
    return {"ok"}
    # return FileResponse('template/index.html')


@app.post("/upload")
async def upload_file(files: List[UploadFile]):
    global filenames
    clear_contents(os.path.join('static', 'inputs'))
    filenames = []
    for file in files:
        filenames.append(file.filename)
        content = file.file._file.read()
        outPath = os.path.join('static', 'inputs', file.filename)
        async with aiofiles.open(outPath, 'wb') as out_file:
            # content = await my_file.file._file.read()  # async read
            await out_file.write(content)  # async write    
    return {"files": len(files)}

@app.get('/extract')
async def process():
    global output
    output = {}
    DATA_DIR = os.path.join('./static', 'inputs')
    OUTPUT_DIR = os.path.join('./static', 'results')
    ERR_DIR = os.path.join('./static', 'failed')
    results = main(DATA_DIR, OUTPUT_DIR, ERR_DIR) 
    # json_str = json.dumps(results, indent=4, default=str, ensure_ascii=False).encode('utf-8')
    for filename in filenames:
        out = results[filename]
        json_str = json.dumps(out, indent=4, default=str, ensure_ascii=False).encode('utf-8')
        output[filename] = json_str
        
    return True

@app.get('/result/{fileNanme}')
async def display(*, fileNanme: str):
    print('ok')
    return Response(content=output[fileNanme], media_type='application/json')
@app.post("/file/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}
@app.post('/ElecBills')
# async def file_upload(my_file: UploadFile = File(...)): # one file.
async def create_files( files: List[UploadFile] = File(...)):
    
    my_file = files[0]
    outPath =  os.path.join('./static', 'inputs', f"{my_file.filename[:-4]}_{str(int(time.time()*100000))}.jpg") 
    content = my_file.file._file.read()
    async with aiofiles.open(outPath, 'wb') as out_file:
        # content = await my_file.file._file.read()  # async read
        await out_file.write(content)  # async write
    
    out = apiMain(outPath)
    os.remove(outPath)
        
    ############## one file
    # async def create_files(file: UploadFile = File(...)):
    #     # # ...
    #     content = files.file._file.read()
    #     outPath = os.path.join('./static', files.filename)
    #     async with aiofiles.open(outPath, 'wb') as out_file:
    #         # content = await my_file.file._file.read()  # async read
    #         await out_file.write(content)  # async write

    #     return {"ok": True}
    ##############
    return out

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
if __name__ == "__main__":
    makedir('./static')
    uvicorn.run("app:app", port=3726, host='0.0.0.0')