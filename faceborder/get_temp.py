# coding: utf-8
# Sample that outputs the value acquired by D6T.
# Please execute the following command before use "pigpio"
#  $ sudo pigpio

import grove_d6t
import pigpio
import time

d6t = grove_d6t.GroveD6t()

def read_temp():
    try:
        print('hi')
        tpn, tptat = d6t.readData()
        print(tpn)
        if tpn == None:
            read_temp()
        else:
            return(float(round(((max(tpn)+2)* 9/5 + 32),1)))
        print(max(tpn))
        print(tpn,"PTAT : %.1f" %tptat)
        return 0
                #time.sleep(1.0)
    except:
        return 0
while 1:
    temp=read_temp()
    print(temp,type(temp))
    time.sleep(1.0)
# def read_temp():
#     return 97.5