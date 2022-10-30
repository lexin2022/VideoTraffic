<<<<<<< HEAD
#  从csv文件中读取5元组serve方向hash值，存入列表a中
#  依次遍历列表a，同HT_TLS逐行匹配，匹配成功时，输出匹配成功的块
import csv

input_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/2021_11_28_12_00_lasts_400s.pcap.FHT.csv'
output_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ht_tls.r_128.csv'
filter_condition_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ipportbHash.csv'

with open(filter_condition_csv) as server_hash_csv, open(output_csv, 'w', newline='') as outp:
    server_hash_list = csv.reader(server_hash_csv)
    for server_hash in server_hash_list:
        csv_o = csv.writer(outp)
        csv_o.writerow([server_hash[0]])
        with open(input_csv, 'r') as inp:
            csv_i = csv.reader(inp)
            for line in csv_i:
                # print(line)
                if len(line) > 7 and line[7] == server_hash[0]:
                    csv_o.writerow(line)
=======
#  从csv文件中读取5元组serve方向hash值，存入列表a中
#  依次遍历列表a，同HT_TLS逐行匹配，匹配成功时，输出匹配成功的块
import csv

input_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/2021_11_28_12_00_lasts_400s.pcap.FHT.csv'
output_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ht_tls.r_128.csv'
filter_condition_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ipportbHash.csv'

with open(filter_condition_csv) as server_hash_csv, open(output_csv, 'w', newline='') as outp:
    server_hash_list = csv.reader(server_hash_csv)
    for server_hash in server_hash_list:
        csv_o = csv.writer(outp)
        csv_o.writerow([server_hash[0]])
        with open(input_csv, 'r') as inp:
            csv_i = csv.reader(inp)
            for line in csv_i:
                # print(line)
                if len(line) > 7 and line[7] == server_hash[0]:
                    csv_o.writerow(line)
>>>>>>> 74d556a (tools)
