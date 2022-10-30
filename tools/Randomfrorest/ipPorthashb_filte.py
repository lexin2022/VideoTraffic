<<<<<<< HEAD
# 基于已确定类型的五元组hash值，为新的一组hash值打标签

import csv

ipPorthashb = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ipportbHash.csv'
video_flow_hash = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ipportbHash_video.csv'
output_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/output.csv'

with open(ipPorthashb) as hash_csv, open(output_csv, 'w', newline='') as outp:
    hash_list = csv.reader(hash_csv)
    for hash in hash_list:
        # print(hash)
        csv_o = csv.writer(outp)
        with open(video_flow_hash, 'r') as video_hash_csv:
            video_hash_list = csv.reader(video_hash_csv)
            for video_hash in video_hash_list:
                if hash[0] == video_hash[0]:
                    csv_o.writerow(hash)
                    continue

# with open(ipPorthashb, 'w', newline='') as hash_csv:
#     hash_reader = csv.reader(hash_csv)
#     for hash_i in hash_reader:
#         with open(video_flow_hash) as video_hash_csv:
#             writer = csv.writer(hash_csv, lineterminator='\n')
#             video_hash_reader = csv.reader(video_hash_csv)
#
#             all = []
#             row = next(hash_reader)
#             for line in video_hash_reder:
#                 if hash_i == line[0]:
#                     row.append(1)
#                     print(row)
#                     all.append(row)
#
#             # writer.writerows(all)

=======
# 基于已确定类型的五元组hash值，为新的一组hash值打标签

import csv

ipPorthashb = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ipportbHash.csv'
video_flow_hash = 'E:/video_traffic_datas/cernet3/cernet-mix_video/ipportbHash_video.csv'
output_csv = 'E:/video_traffic_datas/cernet3/cernet-mix_video/output.csv'

with open(ipPorthashb) as hash_csv, open(output_csv, 'w', newline='') as outp:
    hash_list = csv.reader(hash_csv)
    for hash in hash_list:
        # print(hash)
        csv_o = csv.writer(outp)
        with open(video_flow_hash, 'r') as video_hash_csv:
            video_hash_list = csv.reader(video_hash_csv)
            for video_hash in video_hash_list:
                if hash[0] == video_hash[0]:
                    csv_o.writerow(hash)
                    continue

# with open(ipPorthashb, 'w', newline='') as hash_csv:
#     hash_reader = csv.reader(hash_csv)
#     for hash_i in hash_reader:
#         with open(video_flow_hash) as video_hash_csv:
#             writer = csv.writer(hash_csv, lineterminator='\n')
#             video_hash_reader = csv.reader(video_hash_csv)
#
#             all = []
#             row = next(hash_reader)
#             for line in video_hash_reder:
#                 if hash_i == line[0]:
#                     row.append(1)
#                     print(row)
#                     all.append(row)
#
#             # writer.writerows(all)

>>>>>>> 74d556a (tools)
