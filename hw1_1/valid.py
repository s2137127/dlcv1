
import csv

if __name__ == "__main__":

    cnt = 0
    acc_cnt = 0
    file = open("./output/pred.csv")
    reader = csv.reader(file)
    header = next(reader)

    for a,b in reader:
        # print(a.split('([')[1].split('])')[0],b)
        print(a)
        if a.split('_')[0] == b :
            acc_cnt += 1;
            print(b)
        cnt +=1
    print("acc",acc_cnt/cnt)
    file.close()

