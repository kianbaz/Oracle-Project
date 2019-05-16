import csv
import gzip
# importing readers to convert gzip file of internet traffic
def gzConverter():
    with gzip.open("internetchicago2.gz", "wt", newline="") as file:
        writer = csv.writer(file)


    with gzip.open("internetchicago2.gz", "rt", newline="") as file:
        reader = csv.reader(file)
        print(list(reader))
    return
gzConverter()