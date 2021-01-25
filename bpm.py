import numpy as np
import librosa  # FFmpegのインストールも必要
import os       # ファイル探索用
import csv
import sqlite3

# テーブルを作成
dbname = 'db/track.db'


# 楽曲用データベース
class TrackDB:
    # 初期化
    def __init__(self):
        self.conn = sqlite3.connect(dbname)
        self.cur = self.conn.cursor()

        # テーブルが存在しない場合のみテーブルの作成
        if not self.__check_tracks_table():
            self.__create_tracks_table()

    # 楽曲情報テーブルの作成
    def __create_tracks_table(self):
        self.cur.execute("""
            create table tracks(
            id integer primary key AUTOINCREMENT,
            name string,
            path string,
            bpm float)
            """)
        self.conn.commit()

    # テーブルの存在確認
    def __check_tracks_table(self):
        self.cur.execute("""
            select COUNT(*) from sqlite_master 
            where type ='table' and name='tracks'
            """)
        if self.cur.fetchone()[0] == 0:
            return False
        return True

    # 楽曲の新規追加
    def add_track(self, name, filepath, bpm):
        self.cur.execute('insert into tracks(name, path, bpm) values(?, ?, ?)', [name, filepath, str(bpm)])
        self.conn.commit()

    # 楽曲情報の検索
    def get_track(self, path):
        self.cur.execute('select * from tracks where path = ?', [path])

        track_data = self.cur.fetchone()

        if track_data is None:
            return [], False
        else:
            return track_data, True


# 楽曲用DBの定義
tracks_db = TrackDB()

# 設定値
duration = 30   # 測定対象期間
x_sr = 200      # サンプリング周波数
bpm_min, bpm_max = 60, 240  # BPM範囲


# 楽曲クラス
class Track:
    def __init__(self, filepath):
        # SQLite3に登録されているデータを取得
        track_data, data_exist = tracks_db.get_track(filepath)

        if data_exist:
            self.name = track_data[1]
            self.filepath = track_data[2]
            self.bpm = track_data[3]
        else:
            # 未登録の場合は新規で追加
            self.filepath = filepath
            self.bpm = self.get_bpm()
            file_info = os.path.splitext(filepath)
            self.name = file_info[0]
            tracks_db.add_track(self.name, self.filepath, self.bpm)

    # 楽曲情報の取得
    def get_music_info(self):
        print(self.filepath + "\tBPM: " + str(self.get_bpm(self.filepath)))
        # self.get_beat_time(self.filepath, os.path.splitext(self.filepath)[0])

    # BPM計測関数
    def get_bpm(self):
        # 音源ファイルの読み込み
        y, sr = librosa.load(
            self.filepath,
            offset=0,
            duration=duration,
            mono=True
        )

        # ビート検出用信号の生成
        # リサンプリング & パワー信号の抽出
        x = np.abs(librosa.resample(y, sr, x_sr)) ** 2
        x_len = len(x)  # 200Hz x 30sec = 6,000

        # 各BPMに対応する複素正弦波行列を生成
        M = np.zeros((bpm_max, x_len), dtype=np.complex)

        for bpm in range(bpm_min, bpm_max):
            thete = 2 * np.pi * (bpm / 60) * (np.arange(0, x_len) / x_sr)
            M[bpm] = np.exp(-1j * thete)

        # 各BPMとのマッチング度合い計算 （複素正弦波行列とビート検出用信号との内積）
        x_bpm = np.abs(np.dot(M, x))

        # マッチ度が一番高いBPMを返す
        return np.argmax(x_bpm)

    # ビートの取得
    def get_beat_time(self):
        print('Loading File...')
        y, sr = librosa.load(self.filepath, mono=True)

        print('Loading Beat...')

        # ビートトラッカーでビート発生時のフレームを取得
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # ビートイベント発生をタイムスタンプに
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        print(beat_times)

        print(find_nearest(beat_times, 10.2))
        # CSV出力
        np.savetxt('beat/test.csv', beat_times)


# 特定のディレクトリ内の全ファイルのBPM取得
def load_directory(directory):
    for file in os.listdir(directory):
        # ファイルのパスを作成
        filepath = directory + file

        # ファイルの情報を取得
        file_info = os.path.splitext(filepath)

        # MP3のみ抽出
        if file_info[1] == '.mp3':
            Track(filepath)


# 楽曲の同期
def beat_sync(time_a, time_b):
    pass


# numpy配列から最も近い値を取得
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


load_directory("test/")
