import numpy as np
import librosa  # FFmpegのインストールも必要
import os       # ファイル探索用

# 設定値
duration = 30
x_sr = 200
bpm_min, bpm_max = 60, 240  # BPM範囲


# BPM計測関数
def get_bpm(filepath):
    # 音源ファイルの読み込み
    y, sr = librosa.load(
        filepath,
        offset=38,
        duration=duration,
        mono=True
    )

    # ビート検出用信号の生成
    # リサンプリング & パワー信号の抽出
    x = np.abs(librosa.resample(y, sr, x_sr)) ** 2
    x_len = len(x)

    # 各BPMに対応する複素正弦波行列を生成
    M = np.zeros((bpm_max, x_len), dtype=np.complex)
    for bpm in range(bpm_min, bpm_max):
        thete = 2 * np.pi * (bpm / 60) * (np.arange(0, x_len) / x_sr)
        M[bpm] = np.exp(-1j * thete)

    # 各BPMとのマッチング度合い計算
    # （複素正弦波行列とビート検出用信号との内積）
    x_bpm = np.abs(np.dot(M, x))

    # BPMを算出して返す
    return np.argmax(x_bpm)


# 特定のディレクトリ内の全ファイルのBPM取得
def load_directory(directory):
    for file in os.listdir(directory):
        # ファイルのパスを作成
        file_path = directory + file

        # ファイルの情報を取得
        file_info = os.path.splitext(file_path)

        # MP3のみ抽出
        if file_info[1] == '.mp3':
            print(file_path + "\tBPM: " + str(get_bpm(file_path)))


load_directory("test/")
