import datetime
import csv
import sys

EPOCH = datetime.datetime.utcfromtimestamp(0)
DATA_SET_PATH = './ass3_rawdata/'
RESULT_PATH = 'nanopoch/'


def unix_time_millis(dt):
        return (dt - EPOCH).total_seconds() * 1000.0


def convert_dotnet_tick(ticks):
    _date = datetime.datetime(1, 1, 1) + \
        datetime.timedelta(microseconds=ticks // 10)
    if _date.year < 1900:  # strftime() requires year >= 1900
        _date = _date.replace(year=_date.year + 1900)
    _date = _date + datetime.timedelta(hours=2)
    return _date.strftime("%d.%m.%Y %H:%M:%S:%fZ")[:-3]


def test_dotnet_to_dt():
    epoch = "633790226051280329"
    dt = "27.05.2009 14:03:25:127"

    try:
        print(convert_dotnet_tick(int(epoch)))
        print(dt)
    except:
        print("Missing or invalid argument; use, e.g.:"
              " python ticks.py 636245666750411542")
        print("with result: %s " % convert_dotnet_tick( 636245666750411542))


def test_dt_to_epoch(str_dt):
    dt = datetime.datetime.strptime(str_dt, "%d.%m.%Y %H:%M:%S:%f")

    milis = unix_time_millis(dt)
    nanos = int(milis * 1000000)

    return nanos


def parse_nanopoch(file_name):
    output_file_name = DATA_SET_PATH + RESULT_PATH + file_name

    with open(DATA_SET_PATH + file_name, 'rb') as csv_file:
        output = open(output_file_name, 'wb')
        reader = csv.reader(csv_file, delimiter=',')
        reader.next()  # skip header

        # original header as reference ;)
        # ,id,sensor_id,timestamp,time,x,y,z,activity_label,ankle_l_x,ankle_l_y,ankle_l_z,ankle_r_x,ankle_r_y,ankle_r_z,belt_x,belt_y,belt_z,chest_x,chest_y,chest_z,labelWalking,labelFalling,labelLyingDown,labelLying,labelSittingDown,labelSitting,labelStandingFromLying,labelOnAllFours,labelSittingOnTheGround,labelStandingFromSitting,labelStandingFromSittingOnTheGround

        # write new header
        output.write(",id,sensor_id,timestamps,x,y,z,activity_label,ankle_l_x,ankle_l_y,ankle_l_z,ankle_r_x,ankle_r_y,ankle_r_z,belt_x,belt_y,belt_z,chest_x,chest_y,chest_z,labelWalking,labelFalling,labelLyingDown,labelLying,labelSittingDown,labelSitting,labelStandingFromLying,labelOnAllFours,labelSittingOnTheGround,labelStandingFromSitting,labelStandingFromSittingOnTheGround\n")
        for row in reader:
            # convert time
            raw_dt = row[4]
            timestamp = test_dt_to_epoch(raw_dt)

            new_row = ""
            for i in range(3):
                new_row += row[i] + ","
            new_row += str(timestamp) + ","
            for i in range(5,31):
                new_row += row[i] + ","
            new_row += row[31] + "\n" # final column without comma

            output.write(new_row)

        output.close()
        print("Written " + output_file_name + "!")


def parse_all_files():
    # test_dt_to_epoch()
    # test_dotnet_to_dt()

    files = []
    for person in ['A', 'B', 'C', 'D', 'E']:
        for measure in ['01', '02', '03', '04', '05']:
            files.append(person + measure + '_parsed_raw_data.csv')

    for f in files:
        parse_nanopoch(f)
    print("see you around cowboy.")


def parse_almighty_file():
    parse_nanopoch('parsed_raw_data.csv')
    print("see you around cowgirl.")


if __name__ == '__main__':
    #parse_all_files
    parse_almighty_file()