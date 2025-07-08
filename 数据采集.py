import pyscreenshot as ImageGrab
import time
import cv2

imageFolder = 'captured_images/5/'



for i in range(3):
    time.sleep(2)
    img = ImageGrab.grab(bbox=(399,169,1199,969))
    print('saved.......',i)
    img.save(imageFolder+str(i)+'.png')
    print('clear screen now and redraw now...')
    
    
    