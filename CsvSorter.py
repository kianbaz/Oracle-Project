import csv
import operator
import pandas
import ipaddress

def csvSorter():
    read = open('internetchicago2.csv', 'r')
    csv1 = csv.reader(read, delimiter=',')
    sort = sorted(read, key=operator.itemgetter(3))

    for eachline in sort:
        print(eachline)

    import collections
    data1 = []
    data1 = collections.Counter()
    with open('internetchicago2.csv') as input_file:
        for col in csv.reader(input_file, delimiter=';'):
            data1[col[0]] += 1

    print(data1.most_common())
    return

def ipConverter():
    colnames = ['No.', 'Source', 'Destination', 'Protocol', 'Length', 'Label']
    data = pandas.read_csv('internetchicago2.csv', names = colnames)


    names = data.Destination.tolist()
    ip = data.Destinations.tolist()

    for k, v in ip:
        int(ipaddress.ip_address())
    else:
        print("Converted ip addresses")
    return

csvSorter()
ipConverter()

