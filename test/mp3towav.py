from pydub import AudioSegment

song = AudioSegment.from_mp3("/home/shenxinzhe/Downloads/NAutoAudioFromBramIphone_180904/break.mp3")

song.export("/home/shenxinzhe/Downloads/NAutoAudioFromBramIphone_180904/break.wav", format="wav")


