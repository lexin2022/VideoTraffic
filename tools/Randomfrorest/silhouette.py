#!/usr/bin/python

# ----------------------------------------------------------------------------
#  Silhouette is a real-time, lightweight video classification mechanism that
#  uses only flow statistics (i.e., "shape") for video identification making
#  it payload-agnostic, effective for identifying video flow even when
#  encrypted.
#
#  Used:
#      "Silhouette - Identifying YouTube Video Flows from Encrypted Traffic", 
#       In Proceedings of the 28th ACM International  Workshop on NOSSDAV, 
#       Amsterdam, The Netherlands, June 2018.
#
#  Usage:
#     silhouette.py <pcap file>
#
#  Output format:
#     The output is flow classification results in CSV format,
#      col#1, flow seq in pcap file.
#      col#2, start_tm, relative flow start time, since capture started
#      col#3, flow duration, time elapsed between last seen packet and the first
#             packet in a flow
#      col#4, down link flow volume in KB(excluding IP and Transport headers)
#      col#5, down link flow rate in kbps
#      col#6, avg downlink payload length per packet, excluding IP and Transport
#             headers
#      col#7, number of possible video ADUs
#      col#8, flow type: <video, nonvideo, maybe>
#      col#9, human readable flow id, <proto>:<sip>:<sport>:<dip>:<port>
# ----------------------------------------------------------------------------

import dpkt, struct, time, re, socket
import operator
import os.path
import platform
import hashlib
from pprint import pprint
import sys
from dpkt.ip import IP, IP_PROTO_UDP, IP_PROTO_TCP


# -----------------------------------------------------------------------------
#  generateKey():
#    a utility function to generate human readable flow id. The format as:
#       <protocol>: <srcip>: <srcport>: <dstip>: <dstport>
# -----------------------------------------------------------------------------
# def generateKey(sip, sport, dip, dport=443, proto=IP_PROTO_UDP):
#     key = str(proto) + ":" + str(sip) + ":" + str(sport) + "-" \
#           + str(dip) + ":" + str(dport)
#     return key
def generateKey(sip, sport, dip, dport=443, proto=IP_PROTO_UDP):
    key = str(proto) + "," + str(sip) + "," + str(sport) + "," \
          + str(dip) + "," + str(dport) + ","
    return key


# -----------------------------------------------------------------------------
#  generateId(): (reserved)
#    a utility function to generate a hash key for flow record in k-value
#    store.
# -----------------------------------------------------------------------------
def generateId(key):
    return hashlib.md5(str(key).encode())


# -----------------------------------------------------------------------------
# class definition of Application Data Unit
# -----------------------------------------------------------------------------
class appDataUnit:
    version = 1.0

    def __init__(self, startTime):
        self.start_tm = startTime
        self.data_pkts = 0
        self.data_size = 0
        self.last_active_tm = startTime
        self.stop_tm = startTime
        self.dl_tm = 0

    def setStopTime(self, stopTime):
        self.stop_tm = stopTime

    def countDataPkts(self):
        return self.data_pkts

    def addDataPkt(self, pkt_len, currentTime, retry=False):
        if self.data_pkts == 0:
            self.first_data_pkt_tm = currentTime
        delta = currentTime - self.last_active_tm
        self.last_active_tm = currentTime
        self.data_pkts += 1
        if retry == False:
            self.data_size += pkt_len
        return

    def getDuration(self):
        return (self.stop_tm - self.start_tm)

    def getRate(self):
        duration = self.stop_tm - self.start_tm
        self.dl_tm = self.last_data_pkt_tm - self.start_tm
        if duration > 0:
            rate = self.data_size * 8 / (duration * 1024)
        else:
            rate = 0
        return rate


# -----------------------------------------------------------------------------
# class definition of Flow Record
# -----------------------------------------------------------------------------
class flowRecord:
    version = 1.0
    # traffic rate threshold for video, default as 300Kbps,
    R_v = 300
    # traffic rate threshold for nonvideo/audio, default as 192Kbps,
    R_a = 192
    # pkt length threshold for video, default as 900 B, 2/3 of default MTU=1500B
    L_v = 900
    # pkt length threshold for audio, default as 450 B, 1/3 of default MTU=1500B
    L_a = 450
    # segment length threshold for video ADU, default 100KB
    t_v = 100 * 1024
    # number of ADUs threshold for video, default is 3.
    adu_t = 3
    # pkt length threshold for a request (up stream), default is 500B
    l_req = 500

    def __init__(self, sip, sport, dip, dport=443, proto=IP_PROTO_UDP):
        self.sip = sip
        self.port = sport
        self.dip = dip
        self.dport = dport
        self.proto = proto
        self.key = generateKey(sip, sport, dip, dport, proto)
        self.id = generateId(self.key)
        self.adus = list()
        self.down_total_len = 0
        self.down_pkt_cnt = 0
        self.stopTime = 0
        self.startTime = 0

    def getKey(self):
        return str(self.key)

    def getId(self):
        return str(self.id.hexdigest())

    def setSNI(self, sni):
        self.sni = str(sni)
        return

    def getSNI(self):
        return str(self.sni)

    def addADU(self, adu):
        self.adus.append(adu)
        return

    def setStartTm(self, tm):
        self.startTime = tm
        if self.stopTime == 0:
            self.stopTime = tm
        return

    def setLastTm(self, tm):
        self.stopTime = tm
        return

    def addDownLinkPktLen(self, pktlen):
        self.down_total_len += pktlen
        self.down_pkt_cnt += 1

    def getDownLinkRate(self):
        duration = self.stopTime - self.startTime
        if duration == 0:
            return 0
        rate = self.down_total_len * 8 / (duration * 1024)
        return rate

    def getAvgDownLinkPktLen(self):
        if self.down_pkt_cnt > 0:
            return self.down_total_len / self.down_pkt_cnt
        return 0

    def checkAdus(self):
        if len(self.adus) == 0:
            return
        for adu in self.adus[:]:
            duration = adu.stop_tm - adu.start_tm
            if duration == 0:
                self.adus.remove(adu)
                continue
            if adu.data_size < self.t_v:  # discard ADUs not likely be video segments
                self.adus.remove(adu)
                continue
        return

    def isPaceable(self):
        vol = self.down_total_len
        avg_pktlen = self.getAvgDownLinkPktLen()
        rate = self.getDownLinkRate()
        if (rate >= self.R_v) and (avg_pktlen >= self.L_v) and (len(self.adus) >= self.adu_t):
            # a video flow meets *>all<* of the three conditions.
            isPaceable = 'video'
        elif (vol < self.t_v) or (rate <= self.R_a) or (avg_pktlen <= self.L_a):
            # a non video flow meets *>any<* of the three conditions.
            isPaceable = 'nonvideo'
        else:
            # not enough evidence to decide whether a flow is a video.
            isPaceable = 'maybe'
        return isPaceable

    def toString(self, seq=0, session_start=0, vid="N/A"):
        print("%d," % (seq)),  # seq
        print("%.3lf," % (self.startTime - session_start)),  # start_tm
        print("%.3lf," % (self.stopTime - self.startTime)),  # duration
        print("%.2lf," % (self.down_total_len)),  # vol (KB)
        print("%.2lf," % (self.getDownLinkRate())),  # avgRate (dl) (kbps)
        print("%.2lf," % (self.getAvgDownLinkPktLen())),  # avgPayloadLen(dl) (Bytes)
        print("%d," % (len(self.adus))),  # adu counts
        print("%s," % (self.isPaceable())),  # silhouette result
        print("%s," % (self.key)),  # key
        print("")

    def savestat(self, file, flowType):
        if self.down_pkt_cnt >= 2000:    # 大流过滤
            if flowType == 'video':
                label = '1'
            elif flowType == 'nonvideo':
                label = '0'
            else:
                label = '-1'
            # (rate >= self.R_v) and (avg_pktlen >= self.L_v) and (len(self.adus) >= self.adu_t)
            file.write(self.key + str(self.getDownLinkRate()) + ',' + str(self.getAvgDownLinkPktLen())
                       + ',' + str(len(self.adus)) + ',' + label + '\n')

# ------------------------------------------------------------------------------------------
# main program
# ------------------------------------------------------------------------------------------

# if (len(sys.argv) != 2):
#     print("Usage: ")
#     print(sys.argv[0] + " <pcap file>")
#     sys.exit(-1)

flows = dict()
# pcapFile = sys.argv[1]
# pcapFile = "E:/video_traffic_datas/mix_data/Silhouette-Comparative Experiment/sample-https-Htu3va7yDMg.pcap"
pcapFile = "E:/video_traffic_datas/mix_data/Silhouette-Comparative Experiment/capture_video_flows.pcap"

try:
    # fi = open(sys.argv[1], 'rb')
    fi = open(pcapFile, 'rb')
    pcapin = dpkt.pcap.Reader(fi)

    for ts, buf in pcapin:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except: continue
        try:
            if not isinstance(eth.data, dpkt.ip.IP):    # 判断是否为IP数据包
                continue
            else:
                ip = eth.data
        except: continue

        sip = socket.inet_ntoa(ip.src)
        dip = socket.inet_ntoa(ip.dst)
        proto = ip.p

        if proto == IP_PROTO_UDP:
            try:
                udp = ip.data
                sport = udp.sport
                dport = udp.dport
                payload_len = udp.ulen
            except: continue
        elif proto == IP_PROTO_TCP:
            try:
                tcp = ip.data
                sport = tcp.sport
                dport = tcp.dport
                payload_len = len(tcp.data)
            except: continue
        else:
            continue

        if dport == 443 or dport == 80:
            uplink = True
            key = generateKey(sip, sport, dip, dport, proto)
        elif sport == 443 or sport == 80:
            uplink = False
            key = generateKey(dip, dport, sip, sport, proto)
        else:
            continue

        if key in flows:
            flow = flows[key]
        else:
            if uplink == True:
                flow = flowRecord(sip, sport, dip, dport, proto)
            else:
                flow = flowRecord(dip, dport, sip, sport, proto)
            flows[key] = flow
            flow.setStartTm(ts)
        flow.setLastTm(ts)
        if uplink == False:
            if payload_len > 0:
                flow.addDownLinkPktLen(payload_len)
                if len(flow.adus) > 0:
                    flow.adus[-1].addDataPkt(payload_len, ts)
        else:
            if payload_len > flow.l_req:  # find a new ADU
                newAdu = appDataUnit(ts)
                if len(flow.adus) > 0:
                    flow.adus[-1].setStopTime(ts)
                flow.adus.append(newAdu)

    fi.close()

except IOError as errno:
    print("I/O error {0}".format(errno))

# print out sihoutte results in CSV format
# count = 0
print(len(flows))
strFn = pcapFile + ".csv"
file = open(strFn, 'wt')
file.write('protocol,s_ip,s_port,d_ip,d_port,rate(>300Kb/s),avg_pktlen(>900B),cnt_adu(>3),label\n')   # title
for f in (sorted(flows.values(), key=operator.attrgetter('startTime'))):
    # if count == 0:
    #     session_start = f.startTime
    f.checkAdus()
    flowType = f.isPaceable()
    # f.toString(count, session_start)
    # count += 1
    f.savestat(file, flowType)    # 写比较结果

file.close()
