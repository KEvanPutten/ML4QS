import sys
import csv
import datetime

START = datetime.datetime.strptime('19 06 2018 - 14:17', '%d %m %Y - %H:%M')
EPOCH = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
        return (dt - EPOCH).total_seconds() * 1000.0


def readfile(name, date):
    ret = ''
    last = None
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        reader.next() # skip header
        
        for row in reader:
            # get values
            time_passed = float(row[0])
            lux = float(row[1])

            last = dt = date + datetime.timedelta(seconds=time_passed)
            milis = unix_time_millis(dt)
            nanos = int(milis*1000000)

            ret += '\nlight,smartphone,{},{}'.format(nanos,lux)
    return [ret, last]

files = ['Light.csv']

output = open("parsed/light.csv", 'w');
output.write('sensor_type,device_type,timestamps,illuminance')
date = START
for f in files:
    out = readfile(f, date)
    output.write(out[0])
    date = out[1]
print('bye')
output.close()
