import check_volume as cv
import operator
import numpy as np
error_code_dict ={1: 'The file is not ended with .wav',
                  2: 'The threshold value should be between 0 and 1',
                  3: 'The file should be single channel',
                  4: 'The file is not found',
                  5: 'The duration of file should be less than 30 seconds'}
# print(error_code_dict[1])


test_list = [1, 2, 3]
a, b, c = test_list
# print(a)

file_path = "/home/shenxinzhe/practice/firstrepo/audiofiles/output_unit_test.wav"
threshold_value = 0.04  # 音频中静音部分的阈值
params_test = {'nchannels': 1, 'sampwidth': 2, 'framerate': 44100, 'nframes': 88064, 'comptype': 'NONE', 'compname': 'not compressed'}
params, wave_data = cv.open_autio_file(file_path)
# print(operator.eq(params_test, params))
# np.savetxt("wavedata.txt", wave_data)
# file = open("ut_wavedata.txt", "w")
# for line in wave_data:
#     file.write(str(line) + '\n')
# file.close()

wave_data1 = []
with open("/home/shenxinzhe/practice/firstrepo/test/ut_wavedata.txt") as f:
    i = 0
    for line in f:
        wave_data1.append(int(line.strip()))
        i += 1
wave_data2 = np.array(wave_data1)
print((wave_data == wave_data2).all())
indexs_min, indexs_max, wave_data_without_silence = cv.filter_audio_file(wave_data2, threshold_value)
indexs_min_bm = 70427
indexs_max_bm = 88063
# file = open("ut_wavedata_withoutsilence.txt", "w")
# for line in wave_data_without_silence:
#     file.write(str(line) + '\n')
# file.close()

print('filter done')

frequencies, powers, SPL_result = cv.calculate_freq_domain(wave_data_without_silence, 44100)
# file = open("ut_frequency_data.txt", "w")
# for line in frequencies:
#     file.write(str(line) + '\n')
# file.close()
# file = open("ut_power_data.txt", "w")
# for line in powers:
#     file.write(str(line) + '\n')
# file.close()
print("cal volume done")
