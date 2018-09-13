import check_volume as cv
import time

time_start = time.time()

file_path = "/home/shenxinzhe/practice/firstrepo/audiofiles/output_30s.wav"
threshold_value = 0.3  # 音频中静音部分的阈值

try:
    print("volume (dB) : " + '%f' % cv.check_volume(file_path, threshold_value))
except SystemExit as e:
    print(cv.get_error_message_from_exit_code(e))

time_end = time.time()
print('totally cost', time_end - time_start)
