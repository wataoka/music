import pandas as pd


def csv2txt(csv_file):

    txt_file = csv_file[:-4] + '.txt'
    df = pd.read_csv(csv_file)
    lyric = df['lyric']
    lyric.to_csv(txt_file)



if __name__ == '__main__':

    target_path = './data/あなたに逢えてよかった.csv'
    csv2txt(target_path)

