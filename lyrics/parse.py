# coding: utf-8
import glob
from tqdm import tqdm

def csv2txt(path, csv_file):

    """
    csvファイルをtxtファイルに変換する
    :param path: csvファイルがまとめて置いているディレクトリへのpath
    :param csv_file: 変換したいcsvの名前
    """
    import pandas as pd

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


def json2txt():

    """
    jsonファイルから歌詞を抽出し, txtファイルを生成する.

    :param path: jsonファイルがまとめて置いているディレクトリへのpath
    """

    import json


    file_list = glob.glob('./data/json/*.json')
    for file in file_list:
        f = open(file, 'r')
        j = json.load(f)
        f.close()
        for song in tqdm(j['data']):
            out_file = open('./data/txt/' + song['title'].replace('/', '') + '.txt', 'w')
            out_file.write(song['lyrics'])




def json():
    import json
    file_list = glob.glob('./data/json/*.json')
    for file in file_list:
        f = open(file, 'r')
        j = json.load(f)
        f.close
        for song in j['data']:
            if song['singer'] == '福山雅治':
                print(type(song['lyrics']))
                quit()


if __name__ == '__main__':

    json()

    # file_list = glob.glob('./data/csv/*.csv')
    # for filename in file_list:
    #     filename = filename[11:]
    #     path = './data/csv/'
    #     csv2txt(path, filename)
