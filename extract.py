from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
ocr = PaddleOCR(lang='en', det_model_dir='weights/.paddleocr/det', rec_model_dir='weights/.paddleocr/rec',cls_model_dir='weights/.paddleocr/cls', use_gpu=False) 
def getCoorAndText(results):
    
    yc, xc, text, accu = [], [], [], []
    for res in results:
        if res[1][1] > 0.5: 
            yc.append(int((res[0][1][1]+res[0][2][1])/2))
            xc.append(int((res[0][0][0]+res[0][3][0])/2))
            text.append(res[1][0])
            accu.append(res[1][1])
    
    ycUnique, _ = subset(yc.copy(), 9, 'medi')
    yc = [ycUnique[np.argmin(abs(np.array(ycUnique)-v))] for v in yc]        
        
    yc, xc, text = zip(*sorted(zip(yc, xc, text)))
    newYc, newXc, newText = [], [], []
    try:
        for i in range(len(yc)):
            if i != 0 and yc[i] == y0:
                if not doubleCheck:
                    te = t0 + ' ' + text[i]
                    newText[-1] = te
                    doubleCheck = True
                else: continue
            else:
                newYc.append(yc[i])
                newXc.append(xc[i])
                newText.append(text[i])     
                y0, x0, t0 = yc[i], xc[i], text[i] 
                doubleCheck = False
    except: pass

    return newYc, newXc, newText
def getCategory(text):

    Category, Owner, Address, CategoryInd = '', '', '', None
    for i, tex in enumerate(text):
        if 'residen' in tex.lower():
            Category = 'RESIDENTIAL'
            CategoryInd = i
            try:
                k = i+2 if 'ww' in text[i+1].lower() else i+1
                Owner = text[k]
                for k in range(k+1, i+10):
                    if 'mobil' in text[k].lower(): break
                    Address += ' ' + text[k]
            except: pass
            break
                
        elif 'commer' in tex.lower():
            Category = 'COMMERCIAL'
            CategoryInd = i
            try:
                k = i+2 if 'www' in text[i+1] else i+1
                Owner = text[k]
                for k in range(k+1, i+10):
                    if 'mobil' in text[k].lower(): break
                    Address += ' ' + text[k]
            except: pass
            break
    return Category, Owner, Address, CategoryInd 
def getNameOfBest(text):
    Owner, Address = '', ''
    Owner = text[0]
    Address = ' '.join(text[3:5])
    return Owner, Address
def getNameOfBses(text):

    AccountNo, Owner, Address = '', '', ''
    # get end index
    endIndex = None

    for i in range(len(text)):
        if 'consumer' in text[i].lower() or 'consmer' in text[i].lower():
            endIndex = i
            break                 
    if endIndex is None:
        Owner = text[1].split(':')[-1]
        Owner = Owner.replace('NAME', '')
        Address = ' '.join(text[2:5])
        Address = Address.split(':')[-1].replace('ADDRESS', '')
        try:
            if 'addre' in text[5].lower() or len(text[6]) < 8:
                AccountNos = [v for v in text[6].split() if 'con' in v.lower() and len(v)>7]
                AccountNo = AccountNos[0] if len(AccountNos) > 0 else ""  
            else:
                AccountNos = [v for v in text[5].split() if 'con' in v.lower() and len(v)>7]
                AccountNo = AccountNos[0] if len(AccountNos) > 0 else ""  
            if AccountNo == "": 
                for i in range(3):
                    AccountNos = [v for v in text[5+i].split() if len(v)>7]
                    AccountNo = AccountNos[0] if len(AccountNos) > 0 else "" 
                    if AccountNo != "": break
        except: pass   
    else:
        for i, tex in enumerate(text):

            if 'name' in tex.lower():
                try:
                    Owner = text[i].split(':')[-1]
                    Owner = Owner.replace('NAME', '')
                    Address = ' '.join(text[i+1:i+4])
                    Address = Address.split(':')[-1].replace('ADDRESS', '')
                    AccountNos = [v for v in text[endIndex].split() if 'con' in v.lower() and len(v)>7]
                    AccountNo = AccountNos[0] if len(AccountNos) > 0 else ""

                    break
                except: pass

            if 'addre' in tex.lower():
                try:
                    Owner = text[i-1].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    Address = ' '.join(text[i:i+3])
                    Address = Address.split(':')[-1].replace('ADDRESS', '')
                    AccountNos = [v for v in text[endIndex].split() if 'con' in v.lower() and len(v)>7]
                    AccountNo = AccountNos[0] if len(AccountNos) > 0 else ""

                    break
                except: pass

    return AccountNo, Owner, Address   
def getNameOfTata(text, prop):

    AccountNo, Owner, Address, AccountInd = '', '', '', None
    # get end index
    endIndex = None
    if prop == "right":
        for i in range(len(text)):
            if 'dis.s' in text[i].lower():
                endIndex = i
                break
        for i, tex in enumerate(text):
            if 'consumer' in tex.lower() or 'consmer' in tex.lower():
                try:
                    AccountNos = ''.join(re.findall('\d+', tex))[0:12]
                    AccountNo = AccountNos[0:4] + ' ' + AccountNos[4:8] + ' ' + AccountNos[8:]
                    Owner = text[i+1].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    if endIndex is not None: Address = ' '.join(text[i+2:endIndex])
                    else: Address = ' '.join(text[i+2:i+6])
                    Address = Address.split(':')[-1].replace('Address', '')
                    AccountInd = i
                    break
                except: pass
            if 'name' in tex.lower():
                try:
                    AccountNos = ''.join(re.findall('\d+', text[i-1]))[0:12]
                    AccountNo = AccountNos[0:4] + ' ' + AccountNos[4:8] + ' ' + AccountNos[8:]
                    Owner = text[i].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    if endIndex is not None: Address = ' '.join(text[i+1:endIndex])
                    else: Address = ' '.join(text[i+1:i+5])
                    Address = Address.split(':')[-1].replace('Address', '')
                    AccountInd = i-1
                    break
                except: pass

            if 'addre' in tex.lower():
                try:
                    AccountNos = ''.join(re.findall('\d+', text[i-2]))[0:12]
                    AccountNo = AccountNos[0:4] + ' ' + AccountNos[4:8] + ' ' + AccountNos[8:]
                    Owner = text[i-1].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    if endIndex is not None: Address = ' '.join(text[i:endIndex])
                    else: Address = ' '.join(text[i:i+4])
                    Address = Address.split(':')[-1].replace('Address', '')
                    AccountInd = i-2
                    break  
                except: pass
        
    else:
        for i in range(len(text)):
            if 'consumer' in text[i].lower() or 'consmer' in text[i].lower():
                endIndex = i
                break                 
       
        for i, tex in enumerate(text):
            if 'consumer' in tex.lower() or 'consmer' in tex.lower():
                try:
                    Owner = text[i].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    if endIndex is not None: 
                        Address = ' '.join(text[i+1:endIndex])
                        AccountNos = ''.join(re.findall('\d+', text[endIndex]))[0:12]
                        AccountNo = AccountNos[0:4] + ' ' + AccountNos[4:8] + ' ' + AccountNos[8:]
                    else: 
                        Address = ' '.join(text[i+1:i+5])
                        AccountNos = ""

                    Address = Address.split(':')[-1].replace('Address', '')
                    AccountInd = i
                    break
                except: pass
            if 'name' in tex.lower():
                try:
                    Owner = text[i].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    if endIndex is not None: 
                        Address = ' '.join(text[i+1:endIndex])
                        AccountNos = ''.join(re.findall('\d+', text[endIndex]))[0:12]
                        AccountNo = AccountNos[0:4] + ' ' + AccountNos[4:8] + ' ' + AccountNos[8:]
                    else: 
                        Address = ' '.join(text[i+1:i+5])
                        AccountNos = ""

                    Address = Address.split(':')[-1].replace('Address', '')
                    AccountInd = i
                    break
                except: pass

            if 'addre' in tex.lower():
                try:
                    Owner = text[i-1].split(':')[-1]
                    Owner = Owner.replace('Name', '')
                    if endIndex is not None: 
                        Address = ' '.join(text[i:endIndex])
                        AccountNos = ''.join(re.findall('\d+', text[endIndex]))[0:12]
                        AccountNo = AccountNos[0:4] + ' ' + AccountNos[4:8] + ' ' + AccountNos[8:]
                    else: 
                        Address = ' '.join(text[i+1:i+5])
                        AccountNos = ""

                    Address = Address.split(':')[-1].replace('Address', '')
                    AccountInd = i-1
                    break
                except: pass

    return AccountNo, Owner, Address, AccountInd
        

def subset(set, lim, loc):
    '''
    set: one or multi list or array, lim: size, loc:location(small, medi, large)
    This function reconstructs set according to size of lim in location of loc.
    '''
    cnt, len_set = 0, len(set)        
    v_coor_y1, index_ = [], []
    pop = []
    for i in range(len_set):
        if i < len_set-1:
            try:
                condition = set[i+1][0] - set[i][0]
            except:
                condition = set[i+1] - set[i]
            if condition < lim:
                cnt = cnt + 1
                pop.append(set[i])
            else:
                cnt = cnt + 1
                pop.append(set[i])
                pop = np.asarray(pop)
                try:
                    if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                    else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                except:
                    if loc == "small": v_coor_y1.append(min(pop))
                    elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                    else: v_coor_y1.append(max(pop))  
                index_.append(cnt)
                cnt = 0
                pop = []
        else:
            cnt += 1
            pop.append(set[i])
            pop = np.asarray(pop)
            try:
                if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
            except:
                if loc == "small": v_coor_y1.append(min(pop))
                elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                else: v_coor_y1.append(max(pop))                    
            index_.append(cnt)

    return v_coor_y1, index_    
def parse_month(month_name):
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }
    for key in months.keys():
        if key in month_name.lower(): return months[key]

    return 0  
def extract_dates(text):
    date_formats = [
        r"\d{4}-\d{1,2}-\d{1,2}",     # YYYY-MM-DD
        r"\d{1,2}-\d{1,2}-\d{4}",     # DD-MM-YYYY
        r"\d{1,2}/\d{1,2}/\d{4}",     # DD/MM/YYYY
        r"\d{1,2}-\d{1,2}-\d{2}",     # DD/MM/YY
        r"\d{1,2}/\d{1,2}/\d{2}",     # DD/MM/YY
        r"\d{1,2}\.\d{1,2}\.\d{4}",   # DD.MM.YYYY
        r"\d{1,2}\.\d{1,2}\.\d{2}",   # DD.MM.YY
        r"\d{1,2}\s[a-zA-Z]{3}\s\d{4}",  # DD Mon YYYY
        r"\d{1,2}\s[a-zA-Z]{3,}\s\d{4}",  # D Mon[th] YYYY
        r"\d{1,2}(?:st|nd|rd|th)\sday\s(?:of\ )?[a-zA-Z,]+\s\d{4}",  # Nth day of Month, Year
        r"\d{1,2}(?:st|nd|rd|th)\s[a-zA-Z,]+\s\d{4}",  # Nth Month YYYY
        r"\d{2}:\d{2}",           # HH:MM
        r"\d{2}:\d{2}:\d{2}",     # HH:MM:SS
    ]

    dates = []
    for date_format in date_formats:
        matches = re.findall(date_format, text)
        for match in matches:
            if "-" in match:
                year, month, day = map(int, match.split("-"))

            elif "/" in match:
                day, month, year = map(int, match.split("/"))
            elif "." in match:
                day, month, year = map(int, match.split("."))
            elif " " in match:
                parts = match.split()
                day = int(re.findall('\d+', parts[0])[0])
                month_name = parts[-2]
                month = parse_month(month_name)
                year = int(parts[-1])
            elif ":" in match:
                hour, minute = map(int, match.split(":"))
                # Do something with the time components
            else:
                # Handle other date formats here
                continue
            if year != max(year, month, day): 
                temp = day
                day = year
                year = temp
            dates.append([str(day).zfill(2), str(month).zfill(2), str(year).zfill(4)])
    return dates
def getBillDate(text):
    
    Bill_date = ''
    for i, tex in enumerate(text):
        tex = tex.lower()
        if 'bill' in tex and 'date' in tex:
            tex = re.sub('[a-zA-Z]', '', tex)
            Bill_date = extract_dates(tex)
            if Bill_date == []: 
                return tex
    if Bill_date == '':
        for i, tex in enumerate(text):
            tex = tex.lower()
            if 'bill' in tex or 'date' in tex:
                tex = re.sub('[a-zA-Z]', '', tex)
                Bill_date = extract_dates(tex)
                if Bill_date == []: 
                    return tex            
    if Bill_date == '':
        for i, tex in enumerate(text):
            tex = re.sub('[a-zA-Z]', '', tex).strip()
            Bill_date = extract_dates(tex)
            if Bill_date: break
    if Bill_date != []: Bill_date = '-'.join(Bill_date[0])
    else: Bill_date = ''
    return Bill_date 
def getAccountnNo(text):
    AccountNo = ''
    for i, tex in enumerate(text):
        if 'account' in tex.lower():
            try:
                try: AccountNo = re.findall('\d+', tex)[0]
                except: AccountNo = re.findall('\d+', text[i+1])[0] 
            except: pass
            break  
    return AccountNo  
def getCategoryOfTata(text):
    Category = 'RESIDENTIAL'
    for i, tex in enumerate(text):
        if 'commer' in tex.lower():
            Category = 'COMMERCIAL'
            break
    return Category  
def getCategoryOfBest(text):
    Category, AccountNo = 'RESIDENTIAL', ""

    for i, tex in enumerate(text):
        if 'commer' in tex.lower():
            Category = 'COMMERCIAL'
            break
    for i, tex in enumerate(text):
        AccountNos = [v for v in tex.split() if len(v)>6]
        if len(AccountNos) > 0:
            AccountNo = AccountNos[0]
            break
        
    return Category, AccountNo
def extractFromAdani(img_path, xc, yc):
    output = {}
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]    
    cateRatio = [0.03, 0.04, 0.25, 0.32]
    categoryImg = img[int(cateRatio[0]*h):int(cateRatio[2]*h), int(cateRatio[1]*w):int(cateRatio[3]*w)]
    # cateY = 0.168
    # categoryImg = img[yc+int(0.04*h):yc+int((0.04+cateY)*h), int(cateRatio[1]*w):int(cateRatio[3]*w)]
    cateResults = ocr.ocr(categoryImg, cls=False)
    cate_yc, cate_xc, cateText = getCoorAndText(cateResults[0])
    Category, Owner, Address, CategoryInd = getCategory(cateText)
    
    try: billDateRatio = [0.03+cate_yc[CategoryInd]/h+0.08, 0.52, 0.03+cate_yc[CategoryInd]/h+0.15, 0.79]
    except: billDateRatio = [0.22, 0.52, 0.27, 0.79]
    billImg = img[int(billDateRatio[0]*h):int(billDateRatio[2]*h), int(billDateRatio[1]*w):int(billDateRatio[3]*w)]
    billDateResults = ocr.ocr(billImg, cls=False)
    bill_yc, bill_xc, billText = getCoorAndText(billDateResults[0])
    Bill_Date = getBillDate(billText)
    
    try: accountNoRatio = [0.03+cate_yc[CategoryInd]/h+0.16, 0.15, 0.03+cate_yc[CategoryInd]/h+0.25, 0.38]
    except: accountNoRatio = [0.29, 0.15, 0.37, 0.38]
    accountImg = img[int(accountNoRatio[0]*h):int(accountNoRatio[2]*h), int(accountNoRatio[1]*w):int(accountNoRatio[3]*w)]
    accountNoResults = ocr.ocr(accountImg, cls=False)
    account_yc, account_xc, accountText = getCoorAndText(accountNoResults[0])
    AccountNo = getAccountnNo(accountText)
    
    output['Service_Provider_Name'] = 'adani'
    output['Category'] = Category
    output['Owner_name'] = Owner
    output['Address'] = Address
    output['Bill_Date'] = Bill_Date
    output['Account_no'] = AccountNo
    
    return output

def extractFromReliance(img_path, xc, yc):
    output = {}
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]
    output['Service_Provider_Name'] = 'Reliance'
    output['Category'] = ""
    output['Owner_name'] = ""
    output['Address'] = ""
    output['Bill_Date'] = ""
    output['Account_no'] = ""

    return output 
def extractFromBest(img_path, xc, yc) :
    output = {}
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]    
    ownNameImg = img[yc+110:yc+300, max(0, xc-60):xc+380]
    nameResults = ocr.ocr(ownNameImg, cls=False)
    _, _, Text1 = getCoorAndText(nameResults[0])
    ownerName, address = getNameOfBest(Text1)
    cateImg = img[yc+150:yc+330, xc+830:xc+1020]
    cateResults = ocr.ocr(cateImg, cls=False)
    _, _, Text2 = getCoorAndText(cateResults[0])
    Category, AccountNo = getCategoryOfBest(Text2)
    dateImg = img[yc+430:yc+490, xc+340:xc+520]
    dateResults = ocr.ocr(dateImg, cls=False)
    _, _, date_text = getCoorAndText(dateResults[0])
    Bill_Date = getBillDate(date_text)
    
    output['Service_Provider_Name'] = 'BEST'
    output['Category'] = Category
    output['Owner_name'] = ownerName
    output['Address'] = address
    output['Bill_Date'] = Bill_Date
    output['Account_no'] = AccountNo

    return output     
def extractFromBSES(img_path, xc, yc):
    output = {}
    if xc is None: 
        output['Service_Provider_Name'] = 'BSES'
        output['Category'] = ""
        output['Owner_name'] = ""
        output['Address'] = ""
        output['Bill_Date'] = ""
        output['Account_no'] = ""        
        
        return output
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]    
    # check of date position
    dateCheckImg = img[yc-25:yc+100, xc+50:xc+330]
    dateCheckText = ocr.ocr(dateCheckImg, cls=False)[0]
    billDate, AccountNo, ownerName, address, dateCheck = "", "", "", "", False

    for tex in dateCheckText:
        try:
            x0, x1 = int(min(np.array(tex[0])[:, 0])), int(max(np.array(tex[0])[:, 0]))
            y0, y1 = int(min(np.array(tex[0])[:, 1])), int(max(np.array(tex[0])[:, 1]))
            cx, cy = xc+50+int(x1/2+x0/2), yc-25+int(y1/2+y0/2)            
            dates = extract_dates(tex[1][0]) 
            if len(dates) > 0:
                billDate = '-'.join(dates[0])
                dateCheck = True
                break
        except: pass
    

    if dateCheck:
        consumImg = img[cy-20:cy+20, cx+150:cx+350]
        consumeTexts = ocr.ocr(consumImg, cls=False)[0]
        for tex in consumeTexts:
            if len(tex[1][0]) > 7:
                AccountNo = tex[1][0]
                break
        NameAddressImg = img[cy+30:cy+140, cx+70:cx+500]
        nameAddressTexts = ocr.ocr(NameAddressImg, cls=False)[0]
        yc_1, xc_1, Text_1 = getCoorAndText(nameAddressTexts)
        try:
            ownerName = Text_1[0]
            address = ' '.join(Text_1[1:])
        except: pass
    else:
        NameAddressImg = img[yc+40:yc+240, max(xc-90, 0):xc+460]
        nameAddressTexts = ocr.ocr(NameAddressImg, cls=False)[0]
        yc_1, xc_1, Text_1 = getCoorAndText(nameAddressTexts)
        AccountNo, ownerName, address = getNameOfBses(Text_1)
        BillDateImg = img[yc+50:yc+150, int(w/2):int(w*0.9)]
        BillDateTexts = ocr.ocr(BillDateImg, cls=False)[0]
        yc_1, xc_1, Text_1 = getCoorAndText(BillDateTexts)
        billDate = getBillDate(Text_1)        
        
    output['Service_Provider_Name'] = 'BSES'
    output['Category'] = ""
    output['Owner_name'] = ownerName
    output['Address'] = address
    output['Bill_Date'] = billDate
    output['Account_no'] = AccountNo

    return output    
def extractFromMSEB(img_path, xc, yc):
    output = {}
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]    
    output['Service_Provider_Name'] = 'MSEB'
    output['Category'] = ""
    output['Owner_name'] = ""
    output['Address'] = ""
    output['Bill_Date'] = ""
    output['Account_no'] = ""

    return output  
def extractFromTata(img_path, xc, yc):
    output = {}
    img = cv2.imread(img_path)
    # category: 0.03, 0.04, 0.25, 0.32
    h, w = img.shape[0:2]
    # cateRatio = [0.03, 0.04, 0.25, 0.32]
    # billDateRatio = [0.22, 0.52, 0.27, 0.79]
    # accountNoRatio = [0.29, 0.15, 0.37, 0.38]
    if xc > w/2: Ratio_1, prop = [0, 0, 0.22, 0.5], "right"
    else: Ratio_1, prop = [0, 0.5, 0.15, 1], "left"
    Img_1 = img[int(Ratio_1[0]*h):int(Ratio_1[2]*h), int(Ratio_1[1]*w):int(Ratio_1[3]*w)]
    Results_1 = ocr.ocr(Img_1, cls=False)
    yc_1, xc_1, Text_1 = getCoorAndText(Results_1[0])
    AccountNo, Owner, Address, AccountInd = getNameOfTata(Text_1, prop)
    
    if prop == "right":
        try: billDateRatio = [yc_1[AccountInd]/h+0.14, 0.72, yc_1[AccountInd]/h+0.22, 0.98]
        except: billDateRatio = [0.21, 0.72, 0.27, 0.98]
    else:
        try: billDateRatio = [yc_1[AccountInd]/h+0.084, 0.5, yc_1[AccountInd]/h+0.165, 0.75]
        except: billDateRatio = [0.127, 0.5, 0.2, 0.75]        
    billImg = img[int(billDateRatio[0]*h):int(billDateRatio[2]*h), int(billDateRatio[1]*w):int(billDateRatio[3]*w)]
    billDateResults = ocr.ocr(billImg, cls=False)
    bill_yc, bill_xc, billText = getCoorAndText(billDateResults[0])
    Bill_Date = getBillDate(billText)
    
    if prop == "right":
        try: cateRatio = [yc_1[AccountInd]/h+0.19, 0.72, yc_1[AccountInd]/h+0.25, 0.98]
        except: cateRatio = [0.25, 0.72, 0.3, 0.98]
    else:
        try: cateRatio = [yc_1[AccountInd]/h+0.07, 0.5, yc_1[AccountInd]/h+0.15, 0.75]
        except: cateRatio = [0.107, 0.5, 0.187, 0.75]        
    cateImg = img[int(cateRatio[0]*h):int(cateRatio[2]*h), int(cateRatio[1]*w):int(cateRatio[3]*w)]
    cateoResults = ocr.ocr(cateImg, cls=False)
    cate_yc, cate_xc, cateText = getCoorAndText(cateoResults[0])
    Category = getCategoryOfTata(cateText)
    
    output['Service_Provider_Name'] = 'Tata'
    output['Category'] = Category
    output['Owner_name'] = Owner
    output['Address'] = Address
    output['Bill_Date'] = Bill_Date
    output['Account_no'] = AccountNo
    
    return output
    
    