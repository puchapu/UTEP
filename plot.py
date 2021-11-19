import matplotlib.pyplot as plt
# extract acc from logs
def extract_acc(log_path):
    with open(log_path,"r+") as f:
        a = []
        for line in f.readlines():
            temp = line.strip("\n")
            if "*" in line.strip("\n"):
                a.append(float(temp[9:15]))
    return a[:-1]

# plot acc curve
def plot_acc(acc, epochs):
    # acc: a list of acc, you can get them using extract_acc
    # epochs: total epochs
    x = [i for i in range(1,epochs)]    
    axes = plt.subplots(1, 1, figsize=(5, 4),dpi = 800)
    axes.plot(x,acc,linestyle = "-",color='BLUE',label = "DANN")
    axes.legend(loc = "upper left")
    axes.set_ylabel("Accuracy")
    axes.set_xlabel("Number of epochs")
    axes.set_yticks([50, 60, 70, 80, 90, 100])
    axes.set_xticks([0,5,10,15,20,25,30])
    plt.savefig("./acc.png")

# plot a-distance
def plot_adis(adis,x):
    # adis =[1.9945,1.6262,1.3473]
    # x = ['ResNet', 'DANN', 'DANN+UTEP']
    axes = plt.subplots(1, 1, figsize=(5, 4),dpi=800)
    axes.bar(x[0], adis[0], alpha=0.5, color='red', label = x[0] )
    axes.bar(x[1], adis[1], alpha=0.5, color='blue', label =x[1])
    axes.bar(x[2], adis[2], alpha=0.5, color='green', label = x[2])
    axes.set_ylabel("A-Distance")
    axes.set_xlabel("Transfer Task: A -> W")
    axes.set_ylim(1,2.1)
    axes.set_xticks([])
    axes.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
    axes.legend(loc = "upper right")
    plt.savefig("./a-distance.png")



