import  numpy as np
import sys
import os
import scipy.io.wavfile
import glob
from HTK import HCopy, HTKFile

nfilt = 48

def getFbankFeatures(wavFile, destFile, median, variance, numFrames):
  HCopy('./fbank-48.conf', wavFile, destFile + '.htk')
  htk = HTKFile()
  htk.load(destFile + '.htk')
  fbank = np.asarray(htk.data)
  np.save(destFile + '.npy', fbank)

  num_frames = np.size(fbank[0,:])
  numFrames +=  num_frames
  for i in range(nfilt):
    feat = fbank[i,:].flatten('F')
    median_ut = np.sum(feat)
    variance_ut = np.sum(feat * feat)
    median[i] += median_ut
    variance[i] += variance_ut
  return median, variance, numFrames

def main():
  i = int(sys.argv[1]) # Number class
  kind = sys.argv[2]   # training, development or test
  rootPath = os.getcwd()
  sourceDir = rootPath + '/wav-files/' + kind + '/'
  destDir = rootPath + '/features/fbank/' + kind + '/'
  median_train = np.zeros((nfilt))
  variance_train = np.zeros((nfilt))
  num_frames_train = 0
  srcDir = sourceDir + 'S' + str(i) + '/'
  dstDir = destDir + 'S' + str(i) + '/'
  arr = glob.glob(srcDir + '*.wav')
  for j in arr:
    srcFile = j
    fileName = j.split('/')
    dstFile = dstDir + fileName[len(fileName) - 1].replace('.wav', '')
    [median_train, variance_train, num_frames_train] = getFbankFeatures(srcFile, dstFile, median_train, variance_train, num_frames_train)
  np.save(destDir + 'median-train-' + kind + '-S' + str(i) + '.npy', median_train)
  np.save(destDir + 'variance-train-' + kind + '-S' + str(i) + '.npy', variance_train)
  np.save(destDir + 'num-frames-train-' + kind + '-S' + str(i) + '.npy', num_frames_train)

if __name__ == '__main__':
  main()
