<<<<<<< HEAD
import pandas as pd


if __name__ == "__main__":
    path = "H:/video_traffic_datas/mix_data/facebook_http2_video_720p/video_720p_facebook_mix.pcap.Mixed_rseed-22.pcap.31.13.76.8.pcap.TCPLen.csv"
    save_path = "H:/video_traffic_datas/mix_data/facebook_http2_video_720p/transformation.csv"
    data = pd.read_csv(path)
    length = data.iloc[:, 0].values.tolist()
    amount = data.iloc[:, 1].values.tolist()
    # print(amount)
    list_length = []
    for (i, j) in zip(length, amount):
        for k in range(j):
            list_length.append(i)

    pd.DataFrame(list_length).to_csv(save_path, index=False)

=======
import pandas as pd


if __name__ == "__main__":
    path = "H:/video_traffic_datas/mix_data/facebook_http2_video_720p/video_720p_facebook_mix.pcap.Mixed_rseed-22.pcap.31.13.76.8.pcap.TCPLen.csv"
    save_path = "H:/video_traffic_datas/mix_data/facebook_http2_video_720p/transformation.csv"
    data = pd.read_csv(path)
    length = data.iloc[:, 0].values.tolist()
    amount = data.iloc[:, 1].values.tolist()
    # print(amount)
    list_length = []
    for (i, j) in zip(length, amount):
        for k in range(j):
            list_length.append(i)

    pd.DataFrame(list_length).to_csv(save_path, index=False)

>>>>>>> 74d556a (tools)
