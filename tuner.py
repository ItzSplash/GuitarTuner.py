import numpy as np
import pyaudio


NOTE_MIN = 40       # E3
NOTE_MAX = 64       # E4
FSAMP = 22050       # Sampling frequency in Hz
FRAME_SIZE = 2048   # How many samples per frame?
FRAMES_PER_FFT = 16 # FFT takes average across how many frames?



SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT



NOTE_NAMES = 'E F F# G G# A A# B C C# D D# '.split()



def freq_to_number(f): return 64 + 12*np.log2(f/329.63)
def number_to_freq(n): return 329.63 * 2.0**((n-64)/12.0)
def note_name(n): return NOTE_NAMES[n % NOTE_MIN % len(NOTE_NAMES)] + str(n/12 - 1)



def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP
imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)

stream.start_stream()


window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
print()

while stream.is_active():

    
    buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
    buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE), np.int16)

    # Run the FFT on the windowed buffer
    fft = np.fft.rfft(buf * window)

    # Get frequency of maximum response in range
    freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP

    # Get note number and nearest note
    n = freq_to_number(freq)
    n0 = int(round(n))

    # Console output once we have a full buffer
    num_frames += 1

    if num_frames >= FRAMES_PER_FFT:
        if freq > 170 or freq < 150:
            print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(freq, note_name(n0), n-n0))
        else:
            print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(freq/2, note_name(n0), n-n0))
