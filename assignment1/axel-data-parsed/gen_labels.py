import sys
import csv
import datetime

START = datetime.datetime.strptime('06 06 2018 - 16:10', '%d %m %Y - %H:%M')

EPOCH = datetime.datetime.utcfromtimestamp(0)
def unix_time_millis(dt):
        return (dt - EPOCH).total_seconds() * 1000.0

def readfile(name, date):
    ret = ''
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        reader.next() # skip header
        
        for row in reader:
            # get values
            time_passed = float(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])

            # convert datetime
            # crowdsignals: Time points are expressed in
            #               nanoseconds since the start of time
            #               (which is January 1st 1970 following the UNIX convention).
            # so an epoch with 19 characters...

            last = dt = date + datetime.timedelta(seconds=time_passed)
            milis = unix_time_millis(dt)
            nanos = int(milis*1000000)

            ret += '\naccelerometer,smartphone,{},{},{},{}'.format(nanos,x,y,z)
    return [ret, last]

files = ['sitting/Accelerometer.csv', 'walking/Accelerometer.csv',
         'running/Accelerometer.csv', 'running/Accelerometer.csv']
labels = ['Sitting', 'Walking', 'Running', 'Cycling']

output = open("labels.csv", 'w');
output.write('sensor_type,device_type,label,label_start,label_start_datetime,label_end,label_end_datetime')
date = START
for i in range(0,4):
    begin_dt = date
    begin_nanos = int(unix_time_millis(begin_dt)*1000000)

    out = readfile(files[i], date)
    end_dt = out[1]
    end_nanos = int(unix_time_millis(end_dt)*1000000)

    output.write('\ninterval_label,smartphone,{},{}.000,{},{}.000,{}'.format(labels[i],
            begin_nanos,
            begin_dt.strftime('%d/%m/%Y %H:%M:%S'),
            end_nanos,
            end_dt.strftime('%d/%m/%Y %H:%M:%S')))
    date = end_dt
print('bye')
output.close()
