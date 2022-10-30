<<<<<<< HEAD
import pandas as pd
import math


if __name__ == "__main__":
    path = "C:/Users/13137/Desktop/paper/different flow/video_call_flow_01.pcap.UDPLen.csv"
    save_path = "C:/Users/13137/Desktop/paper/different flow/video_call_flow_01.pck_count.csv"
    data = pd.read_csv(path)
    length = data.iloc[:, 0].values.tolist()
    amount = data.iloc[:, 1].values.tolist()
    list_length = []
    for (i, j) in zip(length, amount):
        for k in range(0, int(j)):
            list_length.append(i)

    pd.DataFrame(list_length).to_csv(save_path, index=False)
=======
import pandas as pd
import math


if __name__ == "__main__":
    path = "C:/Users/13137/Desktop/paper/different flow/video_call_flow_01.pcap.UDPLen.csv"
    save_path = "C:/Users/13137/Desktop/paper/different flow/video_call_flow_01.pck_count.csv"
    data = pd.read_csv(path)
    length = data.iloc[:, 0].values.tolist()
    amount = data.iloc[:, 1].values.tolist()
    list_length = []
    for (i, j) in zip(length, amount):
        for k in range(0, int(j)):
            list_length.append(i)

    pd.DataFrame(list_length).to_csv(save_path, index=False)
>>>>>>> 74d556a (tools)
