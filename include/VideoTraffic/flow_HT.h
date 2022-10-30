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

#ifndef FLOW_HTTPTLS_SET_H
#define FLOW_HTTPTLS_SET_H

#include <vector>
#include "_lib.h/libFlow2SE.h"

class CFlowHT: public IFlow2Object
{
public:
    CFlowHT(uint8_t* buf, int len, IFlow2ObjectCreator* lpFOC, int filter);
    ~CFlowHT();
public:
    bool checkObject();
    bool isSameObject(uint8_t* buf, int len);
public:    
    bool addPacket(CPacket* lppck, bool bSou);
    bool intoElephant(CPacket* lppck);
    bool saveObject(FILE* fp, uint64_t cntP, bool bFin);
public:
    uint32_t getPckCnt() {return cntPck;}
    void incPckCnt() {cntPck++;}
private:
    uint32_t cntPck;
private:
    IFlow2ObjectCreator* lpCreator;
    uint8_t* bufKey;
    int lenKey;
    uint32_t selfHash;
    uint32_t IPport_a_Hash;
    uint32_t IPport_b_Hash;
private:
    std::vector<std::string> vctString;
    int type_HT;
    int flow_filter;
};

//==============================================================================
//==============================================================================
//==============================================================================

class CFlowHTCreator: public IFlow2ObjectCreator
{
public:
    CFlowHTCreator(packet_statistics_object_type type, std::string fname, int filter);
    ~CFlowHTCreator(){}
public:
    IFlow2Object* create_Object(uint8_t* buf, int len);
    int filter_packet(CPacket* lppck);
public:    
    std::string getName() {return strName;}
    packet_statistics_object_type getStatType() {return pso_type;}
    bool isSave() {return true;}
    void beginStat(int num){}
    void endStat(int num){}
private:
    packet_statistics_object_type pso_type;
    std::string strName;
    int FHT_filter;
};

#endif