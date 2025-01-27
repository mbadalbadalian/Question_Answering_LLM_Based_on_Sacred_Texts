import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict
import pandas as pd

#################################################################################################################
### Other Functions

#SaveJSONData Function
def SaveJSONData(json_data,json_filename):
    #Saving data as JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file)
    return

#LoadJSONData Function
def LoadJSONData(json_filename):
    #Loading JSON file into variable
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

#SaveTxtFile Function
def SaveTxtFile(txt_data,txt_filename):
    #Saving txt file
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(txt_data)
    return

#LoadTxtData Function
def LoadTxtData(txt_filename):
    #Loading txt file
    with open(txt_filename,'r') as txt_file:
        json_data = txt_file.read()
    return json_data

#CreateAndSaveTxtFileFromList Function
def CreateAndSaveTxtFileFromList(list_data,txt_filename):
    #Writing a txt file from a list
    with open(txt_filename, 'w') as txt_file:
        for verse,text in list_data.items():
            txt_file.write(f'{verse}: {text}\n')
    return txt_file

#CreateBibleDictionary Function
def CreateBibleDictionary(xml_filename,dictionary_filename):
    #Parse XML content
    xml_parsed_data = ET.parse(xml_filename)
    xml_parsed_data_root = xml_parsed_data.getroot()

    #Initialize dictionary
    ESV_Bible_dict = {}

    #Loop through each book, chapter and verse to create dictionary
    for book in xml_parsed_data_root.findall("b"):
        book_name = book.get("n")
        ESV_Bible_dict[book_name] = {}
        for chapter in book.findall("c"):
            chapter_number = chapter.get("n")
            ESV_Bible_dict[book_name][chapter_number] = {}
            for verse in chapter.findall("v"):
                verse_number = verse.get("n")
                verse_text = verse.text
                ESV_Bible_dict[book_name][chapter_number][verse_number] = verse_text

    #Save dictionary
    SaveJSONData(ESV_Bible_dict,dictionary_filename)
    return ESV_Bible_dict

#ProcessXML Function
def ProcessXML(xml_parsed_data_root):
    output = OrderedDict()

    #Loop through each book, chapter and verse
    for book in xml_parsed_data_root.findall('.//b'):
        book_number = book.get('n')
        for chapter in book.findall('.//c'):
            chapter_number = chapter.get('n')
            for verse in chapter.findall('.//v'):
                verse_number = verse.get('n')
                verse_text = verse.text
                output[f'"{book_number} {chapter_number}:{verse_number}"'] = verse_text
    return output

#CreateBibleText Function
def CreateBibleText(xml_filename,txt_filename):
    ESV_Bible_xml = LoadTxtData(xml_filename)
    ESV_Bible_parsed_xml_root = ET.fromstring(ESV_Bible_xml)
    ESV_Bible_list = ProcessXML(ESV_Bible_parsed_xml_root)
    ESV_Bible_txt = CreateAndSaveTxtFileFromList(ESV_Bible_list,txt_filename)
    return ESV_Bible_txt

#OrderedUniqueList Function
def OrderedUniqueList(original_list):
    seen = set()
    ordered_unique_list = []
    for item in original_list:
        if item not in seen:
            seen.add(item)
            ordered_unique_list.append(item)
    return ordered_unique_list

def LoadDataframe(dataframe_filename):
    dataframe_DF = pd.read_csv(dataframe_filename)
    return dataframe_DF

#CreateBibleDataframe Function
def CreateBibleDataframe(ESV_Bible_txt,ESV_Bible_Book_id_DF_filename,ESV_Bible_DF_filename):
    ESV_Bible_txt = ESV_Bible_txt.split('\n')
    books = []
    chapters = []
    verses = []
    texts = []
    for current_line in ESV_Bible_txt:
        if current_line == '':
            continue
        parts = current_line.split(": ")
        
        if len(parts[0].split(" ")) == 2:
            book = parts[0].split(" ")[0]
        else:            
            first_space_index = parts[0].find(' ')           
            second_space_index = parts[0].find(' ',first_space_index+1)
            book = parts[0][:second_space_index].strip('"')
            
        chapter_verse = parts[0].split(" ")[-1]
        chapter,verse = chapter_verse.split(':')
            
        book = book.strip('"')
        chapter = int(chapter.strip('"'))
        verse = int(verse.strip('"'))
        text = ":".join(parts[1:]).strip()
            
        books.append(book)
        chapters.append(int(chapter))
        verses.append(int(verse))
        texts.append(text)

    unique_books = OrderedUniqueList(books)
    books_number_list = np.arange(1,len(unique_books)+1)

    ESV_Bible_Book_id_DF = pd.DataFrame({'b':unique_books,'id':books_number_list})
    ESV_Bible_DF_original = pd.DataFrame({'b':books,'c':chapters,'v':verses,'Text':texts})
    ESV_Bible_DF = pd.merge(ESV_Bible_Book_id_DF,ESV_Bible_DF_original,on='b',how='right')
    ESV_Bible_DF = ESV_Bible_DF.drop(columns=['b'])
    ESV_Bible_DF.rename(columns={'id': 'b'},inplace=True)
    ESV_Bible_DF['id'] = (ESV_Bible_DF['v'] + ESV_Bible_DF['c']*1e3 + ESV_Bible_DF['b']*1e6).astype(int)
    ESV_Bible_DF = ESV_Bible_DF[['id','b','c','v','Text']]
    ESV_Bible_Book_id_DF.to_csv(ESV_Bible_Book_id_DF_filename,index=False)
    ESV_Bible_DF.to_csv(ESV_Bible_DF_filename,index=False)
    return ESV_Bible_Book_id_DF,ESV_Bible_DF

#CreateOrLoad Function
def CreateOrLoad(xml_filename,dictionary_filename,txt_filename,ESV_Bible_Book_id_DF_filename,ESV_Bible_DF_filename,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        ESV_Bible_dict = CreateBibleDictionary(xml_filename,dictionary_filename)
        ESV_Bible_txt = CreateBibleText(xml_filename,txt_filename)
        ESV_Bible_Book_id_DF,ESV_Bible_DF = CreateBibleDataframe(ESV_Bible_txt,ESV_Bible_Book_id_DF_filename,ESV_Bible_DF_filename)
    else:
        ESV_Bible_dict = LoadJSONData(dictionary_filename)
        ESV_Bible_txt = LoadTxtData(txt_filename)
        ESV_Bible_Book_id_DF = LoadDataframe(ESV_Bible_Book_id_DF_filename)
        ESV_Bible_DF = LoadDataframe(ESV_Bible_DF_filename)
    return ESV_Bible_dict,ESV_Bible_txt,ESV_Bible_Book_id_DF,ESV_Bible_DF

#GetBiblecurrent_lines Function
def GetBiblecurrent_lines():
    Bible_xml_filename = "Initial_Data\\ESVBible_Database.xml"
    Bible_dictionary_filename = "Additional_Data\\ESV_Bible_Dictionary.json"
    Bible_txt_filename = "Additional_Data\\ESV_Bible_Text.txt"
    create_or_load_string = 'Create'
    ESV_Bible_dict,ESV_Bible_txt = CreateOrLoad(Bible_xml_filename,Bible_dictionary_filename,Bible_txt_filename,create_or_load_string)
    return ESV_Bible_dict

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    Bible_xml_filename = "Initial_Data\\ESVBible_Database.xml"
    Bible_dictionary_filename = "Additional_Data\\ESV_Bible_Dictionary.json"
    Bible_txt_filename = "Additional_Data\\ESV_Bible_Text.txt"
    Bible_list_filename = "Additional_Data\\ESV_Bible_List.json"
    ESV_Bible_Book_id_DF_filename = "Additional_Data\\ESV_Bible_Book_id_DF.csv"
    ESV_Bible_DF_filename = "Additional_Data\\ESV_Bible_DF.csv"
    create_or_load_string = 'Load'

    #Main Code
    ESV_Bible_dict,ESV_Bible_txt,ESV_Bible_Book_id_DF,ESV_Bible_DF = CreateOrLoad(Bible_xml_filename,Bible_dictionary_filename,Bible_txt_filename,ESV_Bible_Book_id_DF_filename,ESV_Bible_DF_filename,create_or_load_string)
