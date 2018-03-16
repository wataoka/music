# coding: utf-8
import pandas as pd
import glob

def csv2txt(path, csv_file):

    df = pd.read_csv(path + csv_file)
    lyric = df['lyric']
    lyric.to_csv('tmp.txt')

    file = open('tmp.txt', 'r')

    txt_file = csv_file[:-4] + '.txt'
    out_file = open('./data/txt/' + txt_file, 'w')

    file.readline()
    lines = file.readlines()

    for line in lines:
        line = line.replace('\"', '')
        out_file.write(line)

    file.close()
    out_file.close()







if __name__ == '__main__':

    file_list = glob.glob('./data/csv/*.csv')
    for filename in file_list:
        filename = filename[11:]
        path = './data/csv/'
        csv2txt(path, filename)
