/**
MIT License

Copyright (c) 2021 hwu(hwu@seu.edu.cn), xle(xle@seu.edu.cn)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef VIDEO_FLOWS_SET_H
#define VIDEO_FLOWS_SET_H

#include "_lib.h/libFilterSE.h"
#include <vector>

struct st_pulse
{
    double time;
    double tmData;
    double tmIdle;
};

struct st_big_data
{
    uint8_t protocol;
    uint32_t IP;
    uint16_t port;
    uint8_t label;
};


class CVideoFlows: public IFilterObject
{
public:
    CVideoFlows(uint8_t* buf, int len, IFilterObjectCreator* lpFOC, int num1, int num2, int num3, double tmIdle);
    ~CVideoFlows();
public:     //virtual
    bool checkObject();
    bool isSameObject(uint8_t* buf, int len);
    bool addPacket(CPacket* lppck, bool bSou);
    bool intoElephant(CPacket* lppck);
    bool saveObject(FILE* fp, uint64_t cnt);
public:     //virtual
    uint32_t getPckCnt() {return cntPck;}
    void incPckCnt() {cntPck++;}
private:    //count of packets
    uint32_t cntPck;
private:    //self
    IFilterObjectCreator* lpCreator;
    uint8_t* bufKey;
    int lenKey;
    uint32_t selfHash;

    int stat1, stat2, stat3;
    double time_idle;
private:
    void initial();
    double calLimit(int len);
    bool saveStat(int statnum, int step);
private:    //特征
    double beginTM, endTM, freeTM;
    uint32_t i_pck, i_dp;
    uint32_t o_pck, o_dp;
    uint64_t i_len;
    uint64_t o_len;

    uint32_t cnt_o_area_pck[14];
    double cnt_o_upperlimit_pck;
    uint32_t cnt_i_area_pck[14];
    double cnt_i_upperlimit_pck;

    double tmPulseBegin;
    std::vector <st_pulse> vctPulse;
    bool bPulse;
    int iLabel;
};

class CVFCreator: public IFilterObjectCreator
{
public:
    CVFCreator(packet_statistics_object_type type, std::string fname, int num1, int num2, int num3, double tmIdle, int ratio, int seed);
public:
    IFilterObject* create_Object(uint8_t* buf, int len);
public:    
    std::string getCSVFname() {return strName;}
    packet_statistics_object_type getType() {return pso_type;}
    bool isSave() {return false;}
public:
    bool getBigData(std::string strFN);
    int checkBigDataLabel(uint8_t prot, uint32_t IP, uint16_t port);
private:
    packet_statistics_object_type pso_type;
    std::string strName;
    int statnum1, statnum2, statnum3;
    double tm_idle;
    std::vector <st_big_data> vctBigData;
private:
    void saveTitle(const char* name, int stat);
};

#endif
