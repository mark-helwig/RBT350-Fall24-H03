
import time
import statistics
import serial 


distances = [10., 10., 10., 10., 10.]

arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
time.sleep(2)

def distance():
    if arduino.in_waiting > 0:
        arduino.reset_input_buffer()
        time.sleep(.1)
        data = arduino.readline().decode('utf-8').strip()
        distance = float(data)
    for i in range(len(distances)-1):
        distances[i] = distances[i+1]
        distances[-1] = distance
    
    distance = statistics.mode(distances)
    print(distance)
    return distance/100

# gets the most recent distance point from the object
# def distance():
#   # sets up the sensor and scraps the old data so we only get the newest data on the queue
#   # set Trigger to HIGH
#     GPIO.output(GPIO_TRIGGER, True)
 
#     # set Trigger after 0.01ms to LOW
#     time.sleep(0.00001)
#     GPIO.output(GPIO_TRIGGER, False)
 
#     StartTime = time.time()
#     StopTime = time.time()
 
#     # save StartTime
#     while GPIO.input(GPIO_ECHO) == 0:
#         StartTime = time.time()
 
#     # save time of arrival
#     while GPIO.input(GPIO_ECHO) == 1:
#         StopTime = time.time()
 
#     # time difference between start and arrival
#     TimeElapsed = StopTime - StartTime
#     # multiply with the sonic speed (34300 cm/s)
#     # and divide by 2, because there and back
#     distance = (TimeElapsed * 34300) / 2
    
    # for i in range(len(distances)-1):
    #     distances[i] = distances[i+1]
    #     distances[-1] = distance
    
    # distance = statistics.mode(distances)
 
    # return distance

  
