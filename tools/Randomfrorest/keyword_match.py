<<<<<<< HEAD
import re
import csv
import pandas as pd


tls_regex = "image|images|img|pictures|pic|picture" # 以及pn.或者.pn(n为整数)
get_regex = ".jpeg|.jpg|.tiff|.png|.gif|.psd|.raw|.eps|.svg|.ico"


csv_file = open("H:/cernet3/2021_11_28_08_00_lasts_400s.pcap.FHT.csv")
df = pd.read_csv(csv_file,engine='python')
print(df)

"""
---	---	---							
6	222.190.6.121	25066	58.192.118.131	80	3787704521	3008323833	GET		GET /favicon.ico HTTP/1.1
---	---	---							
6	222.190.6.121	25066	58.192.118.131	80	3787704521	3008323833	GET		GET /favicon.ico HTTP/1.1

"""
list_hash_backword = []
list_info = []
list_label = []
for i in range(len(df)):
    cnt = 0
    if(df[1][i] == "---"):
        list_hash_backword[cnt] = df[7][i+1]
        list_info[cnt] = df[9][i+1]
        if(df[8][i+1] == "TLS"):
            list_label = bool(re.search(tls_regex, list_info[cnt]))
        elif(df[8][i+1] == "GET"):
            list_label = bool(re.search(get_regex, list_info[cnt]))
        cnt = cnt+1

df_match_result = pd.DataFrame(data=None, columns = ["hash_backword", "info", "label"])
df_match_result["hash_backword"] = list_hash_backword
df_match_result["info"] = list_info
=======
import re
import csv
import pandas as pd


tls_regex = "image|images|img|pictures|pic|picture" # 以及pn.或者.pn(n为整数)
get_regex = ".jpeg|.jpg|.tiff|.png|.gif|.psd|.raw|.eps|.svg|.ico"


csv_file = open("H:/cernet3/2021_11_28_08_00_lasts_400s.pcap.FHT.csv")
df = pd.read_csv(csv_file,engine='python')
print(df)

"""
---	---	---							
6	222.190.6.121	25066	58.192.118.131	80	3787704521	3008323833	GET		GET /favicon.ico HTTP/1.1
---	---	---							
6	222.190.6.121	25066	58.192.118.131	80	3787704521	3008323833	GET		GET /favicon.ico HTTP/1.1

"""
list_hash_backword = []
list_info = []
list_label = []
for i in range(len(df)):
    cnt = 0
    if(df[1][i] == "---"):
        list_hash_backword[cnt] = df[7][i+1]
        list_info[cnt] = df[9][i+1]
        if(df[8][i+1] == "TLS"):
            list_label = bool(re.search(tls_regex, list_info[cnt]))
        elif(df[8][i+1] == "GET"):
            list_label = bool(re.search(get_regex, list_info[cnt]))
        cnt = cnt+1

df_match_result = pd.DataFrame(data=None, columns = ["hash_backword", "info", "label"])
df_match_result["hash_backword"] = list_hash_backword
df_match_result["info"] = list_info
>>>>>>> 74d556a (tools)
df_match_result["label"] = list_label