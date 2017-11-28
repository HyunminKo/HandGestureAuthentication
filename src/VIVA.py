import scipy.misc
import random

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("train/train_data.txt") as f:
    for line in f:
        xs.append("train/pos" + line.split()[0])
        with open("train/posGt/"+line.split()[0]) as gf:
            lines=[]
            while True:
                line = gf.readline()
                if not line: break
                if line.startswith('%'): continue
                lines.append(line)
            maxIndex = 0
            areaVal = 0
            for l in range(len(lines)):
                line_arr = lines[l].split(' ')
                if areaVal < float(line_arr[3])*float(line_arr[4]):
                    areaVal = float(line_arr[3])*float(line_arr[4])
                    maxIndex = l
            line_arr = lines[maxIndex].split(' ')
            line_arr[1] = round(float(line_arr[1])/3.6363,1)
            line_arr[2] = round(float(line_arr[2])/4.8,1)
            line_arr[3] = round(float(line_arr[3])/3.6363,1)
            line_arr[4] = round(float(line_arr[4])/4.8,1)
            ys.append([line_arr[1],line_arr[2],line_arr[3],line_arr[4]])

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]), [176, 100]))
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]), [176, 100]))
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
