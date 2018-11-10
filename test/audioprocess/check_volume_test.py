import check_volume as cv
import numpy as np
import unittest


class CheckVolumeTest(unittest.TestCase):
    file_path = "/home/shenxinzhe/practice/firstrepo/audiofiles/output_unit_test.wav"
    threshold_value = 0.04  # 音频中静音部分的阈值

    def test_check_volume(self):
        volume = cv.check_volume(self.file_path, self.threshold_value)
        self.assertEqual(round(volume, 8), 50.69395405)

    def test_open_autio_file(self):
        wave_data_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_wavedata.txt") as f:
            for line in f:
                wave_data_benchmark.append(int(line.strip()))

        params_list_benchmark = [1, 2, 44100, 88064, 'NONE', 'not compressed']

        params, wave_data = cv.open_autio_file(self.file_path)
        self.assertEqual(wave_data.all(), np.array(wave_data_benchmark).all())
        self.assertEqual(list(params), params_list_benchmark)

    def test_filter_audio_file(self):
        wave_data_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_wavedata.txt") as f:
            for line in f:
                wave_data_benchmark.append(int(line.strip()))

        wave_data_without_silence_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_wavedata_withoutsilence.txt") as f:
            for line in f:
                wave_data_without_silence_benchmark.append(int(line.strip()))

        indexs_min, indexs_max, wave_data_without_silence = \
            cv.filter_audio_file(np.array(wave_data_benchmark), self.threshold_value)

        self.assertEqual(wave_data_without_silence.all(), np.array(wave_data_without_silence_benchmark).all())
        self.assertEqual(indexs_min, 70427)
        self.assertEqual(indexs_max, 88063)

    def test_generate_audio_file(self):
        wave_data_without_silence_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_wavedata_withoutsilence.txt") as f:
            for line in f:
                wave_data_without_silence_benchmark.append(int(line.strip()))
        params = [1, 2, 44100]
        cv.generate_audio_file(params, np.array(wave_data_without_silence_benchmark).tostring(),
                               self.file_path, "_ut")

    def test_calculate_freq_domain(self):
        wave_data_without_silence_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_wavedata_withoutsilence.txt") as f:
            for line in f:
                wave_data_without_silence_benchmark.append(int(line.strip()))

        frequency_data_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_frequency_data.txt") as f:
            for line in f:
                frequency_data_benchmark.append(float(line.strip()))

        power_data_benchmark = []
        with open("/home/shenxinzhe/practice/firstrepo/test/ut_power_data.txt") as f:
            for line in f:
                power_data_benchmark.append(float(line.strip()))

        frame_rate = 44100
        frequencies, powers, SPL_result = \
            cv.calculate_freq_domain(np.array(wave_data_without_silence_benchmark), frame_rate)
        self.assertEqual(round(SPL_result, 8), 50.69395405)
        self.assertEqual(frequencies.all(), np.array(frequency_data_benchmark).all())
        self.assertEqual(powers, power_data_benchmark)

    def test_calculate_weight_A(self):
        freq = 125
        result = cv.calculate_weight_A(freq)
        self.assertEqual(round(result, 8), -16.18965248)


if __name__ == '__main__':
    unittest.main()