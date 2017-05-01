import json
import numpy as np


def load_data(path,numOfPackages="ALL"):
    dataArray=[]
    index = 1
    with open(path) as file:
        for line in file:
            jsonLine = json.loads(line)
            dataArray.append(jsonLine)
            if numOfPackages != "ALL":
                if index == int(numOfPackages)*25000:
                    break
            index=index+1
    dataArray = np.array(dataArray)
    print("json Loaded")
    return dataArray
