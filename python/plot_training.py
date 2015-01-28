import pylab as pl
import time

log_path = 'caffe/64_sp/lr0.1b256/lr0.1b256.log'
end = False
pl.ion()
fig = pl.figure()
while True:
    f = open(log_path, 'rb')
    #code to parse data and plot four charts
    test_loss, train_loss, test_acc, it = [], [], [], []
    for l in f:
        line = l.split() + [None] * 9
        if line[6] == 'Testing':
            it.append(int(line[5].strip(',')))
        elif line[8] == 'accuracy':
            test_acc.append(line[10])
        elif line[7] == '#1:':
            test_loss.append(line[10])
        elif line[4] == 'Train':
            train_loss.append(line[10])
        elif line[5] == 'Done.':
            end = True
    print len(it), it
    print len(test_acc), test_acc
    print len(test_loss), test_acc
    print len(train_loss), test_acc

    ax = pl.plot(it, test_loss, color='blue')
    ax = pl.plot(it, train_loss, color='red')
    ax = pl.plot(it, test_acc[1:], color='green')
    if not end:
        pl.draw()
        time.sleep(30) 
    else:
        break
pl.show()
