import pytesseract
import cv2
from pytesseract import Output
import re
import matplotlib.pyplot as plt
import os
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from libs.constants import DEFAULT_ENCODING
from libs.ustr import ustr
XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'



class LabelFile:
    def __init__(self):
        self.x2min_lst = []
        self.y2min_lst = []
        self.x2max_lst = []
        self.y2max_lst = []
    
    def getTesseractData(self,image):
        """This method gets the raw data of image in dictionary"""
        
        self.image = image
        data = pytesseract.image_to_data(image, config = ('-l eng --oem 1 --psm 6'),output_type=Output.DICT)
        datalist = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            datalist.append(data['text'][i].strip())
        return datalist,data
    
    def findCordinates(self,index,data):
        """Method finds the co-ordinates using the index and raw data we got out of getTesseractData"""
        
        (x,y,w,h) = (data['left'][index],data['top'][index],data['width'][index],data['height'][index])
        xmin = x-10
        ymin = y-10
        xmax = x+w+10
        ymax = y+h+10
#       cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#       plt.imshow(self.image)
#       plt.show()
        return xmin,ymin,xmax,ymax
    
    def getCordinates(self,datalist,inputDict,data):
        
        '''returns the cordinates of each object '''
        
        index = datalist.index(inputDict['MemberId'])
        x1min,y1min,x1max,y1max = self.findCordinates(index,data)
        Name = "MemberId"
        self.writer.addBndBox(x1min,y1min,x1max,y1max,Name,'0')
        
        matches = re.finditer(' ',inputDict['PayerName'] )
        matches_positions = [match.start() for match in matches]
        size = len(matches_positions)
        if size==0:
            index = datalist.index(inputDict['PayerName'])
            x2min,y2min,x2max,y2max = self.findCordinates(index,data)
            Name = inputDict['PayerName']
            self.writer.addBndBox(x2min,y2min,x2max,y2max,Name,'0')
        else:
            matches_positions.append(len(inputDict['PayerName']))
            size+=1
            start = -1
            i = 0
            while size>0:
                size-=1
                end = matches_positions[i]
                index = datalist.index(inputDict['PayerName'][start+1:end])
                x2min,y2min,x2max,y2max = self.findCordinates(index,data)
                self.x2min_lst.append(x2min)
                self.y2min_lst.append(y2min)
                self.x2max_lst.append(x2max)
                self.y2max_lst.append(y2max)
                start = end
                i+=1
            x2min = min(self.x2min_lst)
            y2min = min(self.y2min_lst)
            x2max = max(self.x2max_lst)
            y2max = max(self.y2max_lst)
            Name = inputDict['PayerName']
            self.writer.addBndBox(x2min,y2min,x2max,y2max,Name,'0')
        return
            
    def saveVocFile(self,imagePath,userInputDict):
        """Method gets the required data to create the .xml file  which we will use later for trainig using YOLOv3 or any other
        obje detection algorithms"""
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        imageShape = image.shape
        self.writer = PascalVocWriter(imgFolderName,imgFileName,imageShape,localImgPath=imagePath)
        datalist,data = self.getTesseractData(image)
        self.getCordinates(datalist,userInputDict,data)
        self.writer.save()
        return

class PascalVocWriter:
    
    '''Init function, gets invoked on class instance creation'''
        
    def __init__(self,folderName,fileName,imageSize,databaseSrc='Unknown',localImgPath=None):    
        self.foldername = folderName
        self.filename = fileName
        self.databaseSrc = databaseSrc
        self.imgSize = imageSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False
        self.annotationfileName = os.path.splitext(fileName)[0]
        self.annotationsFolderPath = os.path.join("Annotations",self.annotationfileName )
        
        
    def addBndBox(self,xmin,ymin,xmax,ymax,name,difficult):
        """
        saves the bounding box of each object
        """
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)
            
            
    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
            # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """Createsthe basic Xml file with proper tags and data"""
        
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def appendObjects(self, top):
        """Method append the object, objects is nothing but fieldfor which we got the co-ordinates
        in my example it's payerName and MemberId"""
        
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
        return
        
    def save(self, targetFile = None):
        """Saves the file after performing the prettify - that's nothing but beautfying the xml file with proper indentation"""
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.annotationsFolderPath + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)
        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()
        return
		
		

"""Creating an instance of LabelFile and passing the objects for which we are trying to get the co-ordinates"""

# input dictionary of the field for which you want to create the annotation file
inputInputDict = {"MemberId" : "593-05-5119-A", "PayerName": "MEDICARE HEALTH INSURANCE"}

#it's path of an image
imagePath = './images/503_593055119A.jpg' 
label =  LabelFile()
label.saveVocFile(imagePath,inputInputDict)







