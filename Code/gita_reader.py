#Reads the Bhagwad Gita csv file and writes certain columns of important information into another one
import csv

#A function meant to extract the columns from input file and place it into the output file
def extract_columns(input_file, output_file):
    columns = [0, 1, 6]  # Columns we are interested in
    column_names = ['c', 'v', 'Text']  # New column names

    #Open and read all contents of the specified csv file
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        read = csv.reader(csv_file)
        data = [row for row in read]

    #extracts row data of all columns we are interested in
    data = [[row[i] for i in columns] for row in data]

    #Open and write the extracted data into the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as new_csv_file:
        write = csv.writer(new_csv_file)
        write.writerow(column_names)  # Writing the new column names
        write.writerows(data[1:])

# Example usage:
input_file = 'Additional_Data/Bhagwad_Gita_Verses_English.csv' #The input filename
output_file = 'Additional_Data/Gita_DF.csv'  #The output filename
extract_columns(input_file, output_file) #Extract the required columns from the input