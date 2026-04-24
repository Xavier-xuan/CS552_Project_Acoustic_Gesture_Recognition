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

import argparse, math, wave, struct, scipy, numpy
'''
parser = argparse.ArgumentParser(description='Generate a sine wave.')

parser.add_argument('-s', action='store_true', help='set stereo mode; if missing, the file will be saved in mono', default=False, dest='stereo')
parser.add_argument('-t', action='store', type=float, help='set sine wave\'s duration in seconds', default=10.0, dest='duration')
parser.add_argument('-f', action='store', type=float, help='set sine wave\'s frequency [0,20000]Hz', default=400.0, dest='frequency')
parser.add_argument('-v', action='store', type=float, help='set sine wave\'s amplitude [1,10]', default=10, dest='volume')
parser.add_argument('-o', action='store', help='set name of wav file', default='chirp.wav', dest='output')

args = parser.parse_args()
'''
IS_STEREO = True
SAMPLE_RATE = 48000.0 		# hertz
NUM_SECONDS = 0.5  	# seconds
FREQUENCY = 12000    	# hertz
VOLUME = 10 * 100
OUTPUT_FILE = 'chirp.wav' 	# filepath

assert NUM_SECONDS > 0.0, 'Duration must be higher than 0 seconds.'
#assert 0 <= FREQUENCY <= 20000.0, 'Wave frequency must be positive and lesser than 20000 Hz.'
assert 100 <= VOLUME <= 1000.0, 'Volume must be higher than 0 and lesser than 100.'

'''log = (
	'Generating a cosine wave.\n\tSample rate: '+str(SAMPLE_RATE)+
	' Hz\n\tDuration: '+ str(NUM_SECONDS)+
	' s\n\tFrequency: '+str(FREQUENCY)+ 
	' Hz\n\tStereo: '+str(IS_STEREO)+
	'\n\tVolume: '+str(VOLUME)+
	'\n\tDestination: '+str(OUTPUT_FILE)
	)

print (log)'''

file = wave.open(OUTPUT_FILE,'wb')
file.setnchannels(2 if IS_STEREO else 1)
file.setsampwidth(2) 
file.setframerate(SAMPLE_RATE)

half_sec_samples = int(0.5 * SAMPLE_RATE)

samps = numpy.arange(half_sec_samples) / SAMPLE_RATE

vals = scipy.signal.chirp(samps, f0=12000, t1=0.5, f1=15000) * VOLUME

for i in range(int(0.5 * SAMPLE_RATE)):
	loc_val = int(vals[i]) 
	data = struct.pack('<hh', loc_val, loc_val) if IS_STEREO else struct.pack('<h', loc_val)
	file.writeframesraw(data)


file.close()