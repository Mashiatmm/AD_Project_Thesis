import pandas as pd

df=pd.read_csv("ADNIMERGE.csv")
bl_types=df['DX_bl'].unique()
dxs=df['DX'].unique()
transitions={}
for b in bl_types:
    for d in dxs:
        transitions[(b,d)]=0
for i in range(df.shape[0]):
    for b in bl_types:
        for d in dxs:
            if(df['DX_bl'][i]==b and df['DX'][i]==d):
                transitions[(b,d)]+=1

# create new statistics csv file
stat=open("statistics_of_adnimerge.csv","w")
stat.write("DX_bl,Dx,Number of transitions\n")


for b in bl_types:
    for d in dxs:
        print("Transition from",b,"to",d,"is",transitions[(b,d)])
        stat.write(str(b)+","+str(d)+","+str(transitions[(b,d)])+"\n")

stat.close()