import pickle

path = 'C:/Users/bandi/Downloads/CS584/HW1/'

file = open(path + 'vector', 'rb')
vector = pickle.load(file)
print(len(vector))
val_list = []
for i in vector:

    if i[0][0] > 1.0:
        val_list.append(i[0][1])

    else:
        val0 = float((i[0][0]) * int(i[0][1])) * (0.4)
        val1 = float((i[1][0]) * int(i[1][1])) * (0.2)
        val2 = float((i[2][0]) * int(i[2][1])) * (0.2)
        val3 = float((i[3][0]) * int(i[3][1])) * (0.1)
        val4 = float((i[4][0]) * int(i[4][1])) * (0.1)

        print(val0, (i[0][0]), (i[0][1]))
        print(val1, (i[1][0]), (i[1][1]))
        print(val2, (i[2][0]), (i[2][1]))
        print(val3, (i[3][0]), (i[3][1]))
        print(val4, (i[4][0]), (i[4][1]))

        final_val = val0 + val1 + val2 + val3 + val4
        if (final_val > 0):
            val_list.append(1)
        else:
            val_list.append(-1)
        print("\n")

# for i in val_list:
#     print(i)

filename = path + "format.dat"
with open(filename, 'w') as filehandle:
    for listitem in val_list:
        filehandle.write('%s\n' % listitem)
