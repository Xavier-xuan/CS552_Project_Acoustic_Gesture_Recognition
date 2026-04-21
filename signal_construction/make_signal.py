# #MIT License

# Copyright (c) [2018] [Alessandro Cudazzo, Francesco Capuzzolo]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse, math, wave, struct, scipy, numpy as np

# This script has been modified to produce 1 specific type of signal, and thus does not resemble the original script in form.

parser = argparse.ArgumentParser(description='Generate a sine wave.')

parser.add_argument('-s', action='store_true', help='set stereo mode; if missing, the file will be saved in mono', default=False, dest='stereo')
parser.add_argument('-t', action='store', type=float, help='set sine wave\'s duration in seconds', default=5.0, dest='duration')
parser.add_argument('-f', action='store', type=float, help='set sine wave\'s frequency [0,20000]Hz', default=400.0, dest='frequency')
parser.add_argument('-v', action='store', type=float, help='set sine wave\'s amplitude [1,10]', default=10, dest='volume')
parser.add_argument('-o', action='store', help='set name of wav file', default='sine_wave.wav', dest='output')

args = parser.parse_args()

IS_STEREO = True
SAMPLE_RATE = 48000.0 		# hertz
#NUM_SECONDS = args.duration  	# seconds
#FREQUENCY = args.frequency    	# hertz
VOLUME = args.volume * 100
OUTPUT_FILE = args.output 	# filepath


FREQ_GAP = 350
FREQ_COUNT = 16
TOTAL_TRIALS = 10

#assert NUM_SECONDS > 0.0, 'Duration must be higher than 0 seconds.'
#assert 0 <= FREQUENCY <= 20000.0, 'Wave frequency must be positive and lesser than 20000 Hz.'
assert 100 <= VOLUME <= 1000.0, 'Volume must be higher than 0 and lesser than 100.'

file = wave.open(OUTPUT_FILE,'wb')
file.setnchannels(2 if IS_STEREO else 1)
file.setsampwidth(2) 
file.setframerate(SAMPLE_RATE)

#generate chirp
half_sec_samples = int(0.5 * SAMPLE_RATE)
samps = np.arange(half_sec_samples) / SAMPLE_RATE
vals = scipy.signal.chirp(samps, f0=12000, t1=0.5, f1=15000) * VOLUME

for i in range(int(1 * SAMPLE_RATE)):
		data = struct.pack('<hh', 0, 0) if IS_STEREO else struct.pack('<h', 0)
		file.writeframesraw(data)

for i in range(int(0.5 * SAMPLE_RATE)):
		value = int(VOLUME * math.sin(2 * 3000 * math.pi * float(i) / float(SAMPLE_RATE)))
		data = struct.pack('<hh', value, 0) if IS_STEREO else struct.pack('<h', value)
		file.writeframesraw(data)

for i in range(int(0.5 * SAMPLE_RATE)):
		value = int(VOLUME * math.sin(2 * 5000 * math.pi * float(i) / float(SAMPLE_RATE)))
		data = struct.pack('<hh', 0, value) if IS_STEREO else struct.pack('<h', value)
		file.writeframesraw(data)

for i in range(TOTAL_TRIALS):
	print(f"Current reps: {i}")

	#1.5 secs Silence
	for i in range(int(1.5 * SAMPLE_RATE)):
		data = struct.pack('<hh', 0, 0) if IS_STEREO else struct.pack('<h', 0)
		file.writeframesraw(data)

	#0.5 secs 12kHz to 15kHz Chirp
	for i in range(int(0.5 * SAMPLE_RATE)):
		val = int(vals[i])
		data = struct.pack('<hh', 0, val) if IS_STEREO else struct.pack('<h', val)
		file.writeframesraw(data)

	#0.5 secs Silence
	for i in range(int(0.5 * SAMPLE_RATE)):
		data = struct.pack('<hh', 0, 0) if IS_STEREO else struct.pack('<h', 0)
		file.writeframesraw(data)

	#3 secs 17-23 freq, 
	for i in range(int(3 * SAMPLE_RATE)):
		value = 0.0

		#0.2 sec 440Hz beep (1 sec in)
		if (i >= int(0.8 * SAMPLE_RATE) and i < int(1 * SAMPLE_RATE)):
			value += int(VOLUME * math.sin(2 * 440 * math.pi * float(i) / float(SAMPLE_RATE)))

		for j in range(FREQ_COUNT):
			value += VOLUME * math.cos(2 * (17000 + FREQ_GAP*j) * math.pi * float(i) / float(SAMPLE_RATE))
		data = struct.pack('<hh', 0, int(value)) if IS_STEREO else struct.pack('<h', int(value))
		file.writeframesraw(data)

	#0.2 secs 660Hz beep
	for i in range(int(0.2 * SAMPLE_RATE)):
		value = int(VOLUME * math.sin(2 * 660 * math.pi * float(i) / float(SAMPLE_RATE)))
		data = struct.pack('<hh', 0, value) if IS_STEREO else struct.pack('<h', value)
		file.writeframesraw(data)

file.close()