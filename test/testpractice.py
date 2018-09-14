import sys
sys.path.append("/home/shenxinzhe/practice/firstrepo/audioprocess/check_volume.py")


error_code_dict ={1: 'The file is not ended with .wav',
                  2: 'The threshold value should be between 0 and 1',
                  3: 'The file should be single channel',
                  4: 'The file is not found',
                  5: 'The duration of file should be less than 30 seconds'}
print(error_code_dict[1])


test_list = [1, 2, 3]
a, b, c = test_list
print(a)

file_path = "/home/shenxinzhe/practice/firstrepo/audiofiles/output_25.wav"
threshold_value = 0.3  # 音频中静音部分的阈值

params, wave_data = open_autio_file(file_path)
