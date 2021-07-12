import os
import pandas as pd
import matplotlib.pyplot as plt

if not os.path.exists("charts"):
    os.makedirs("charts")

df=pd.read_csv('resultsmnist/results_mae_mnist_fashion_0_6_2021-07-12_09-54-41.csv',index_col=0)
df=df.rename(columns={"cc_quapy": "CC","pcc_quapy":"PCC","ac_quapy":"AC","pac_quapy":"PAC","hdy":"HDy","quanet":"QuaNet","dqn":"DQN-MAX","histnet":"HistNet","histnetc":"HistNetC"})
df=df.drop(['mean'])
myFig=plt.figure()
boxplot = df.boxplot(column=['CC', 'PCC', 'AC','PAC','HDy','QuaNet','DQN-MAX','HistNet','HistNetC'],rot=90)
myFig.savefig("charts/mnist.svg", format="svg",bbox_inches = "tight")


df=pd.read_csv('resultsimdb/results_mae_imdb_2021-07-12_13-33-32.csv',index_col=0)
df=df.rename(columns={"cc_quapy": "CC","pcc_quapy":"PCC","ac_quapy":"AC","pac_quapy":"PAC","hdy_quapy":"HDy","quanet":"QuaNet","dqn":"DQN-MAX","histnet":"HistNet","histnetc":"HistNetC"})
df=df.drop(['mean'])
myFig=plt.figure()
boxplot = df.boxplot(column=['CC', 'PCC', 'AC','PAC','HDy','QuaNet','DQN-MAX','HistNet','HistNetC'],rot=90)
myFig.savefig("charts/imdb.svg", format="svg",bbox_inches = "tight")