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

#include "VideoTraffic/flow_HT.h"
#include "_lib.h/libHashSE.h"
#include "_TLS/TLS_tools.h"
#include <iostream>

using namespace std;

CFlowHT::CFlowHT(uint8_t* buf, int len, IFlow2ObjectCreator* lpFOC, int filter)
{
    cntPck = 0;
    lpCreator = lpFOC;
    if(len>0)
    {
        lenKey = len;
        bufKey = (uint8_t*)calloc(lenKey, sizeof(uint8_t));
        if(bufKey)
        {
            memcpy(bufKey, buf, len);
           	selfHash = CHashTools::HashBuffer(bufKey, lenKey, 32);
        }
    }
    type_HT = 0;
    flow_filter = filter;
    IPport_a_Hash = IPport_b_Hash = 0;
}

CFlowHT::~CFlowHT()
{
    if(bufKey)
        free(bufKey);
    vctString.clear();
}

bool CFlowHT::checkObject()
{
    if(lenKey>0 && bufKey)
        return true;
    else
        return false;
}

bool CFlowHT::isSameObject(uint8_t* buf, int len)
{
    bool bout = false;

    if(lenKey == len)
    {
        if(memcmp(bufKey, buf, len)==0)
            bout = true;
    }
    return bout;
}

bool CFlowHT::addPacket(CPacket* lppck, bool bSou)
{
    uint8_t bufByte[UINT8_MAX];
    int len;
    bool bBuf;

    if(lppck && lppck->getLenPayload()>0)
    {
        if(lppck->getPckNum()==8)
            int wos = 1;
        if(lppck->getProtocol()==6)
        {
            int len;
            uint8_t* buf = lppck->getPacketPayload(len);
            if(len>20 && buf[0]=='G' && buf[1]=='E' && buf[2]=='T' && buf[3]==' ')
            {
                char* bufpos = strstr((char*)buf, "\r\n");
                if(bufpos)
                {
                    if(IPport_a_Hash==0 && bSou)
                    {
                        bBuf = CPacketTools::getHashBuf_IPport(lppck, bufByte, len, true);
                        if(bBuf)
                           	IPport_a_Hash = CHashTools::HashBuffer(bufByte, len, 32);
                        bBuf = CPacketTools::getHashBuf_IPport(lppck, bufByte, len, false);
                        if(bBuf)
                           	IPport_b_Hash = CHashTools::HashBuffer(bufByte, len, 32);
                        type_HT = 1;
                    }
                    bufpos[0] = 0;
                    string strGET = (char*)buf;
                    vctString.push_back(strGET);
                }
            }
            else
            {
                char bufSNI[UINT8_MAX] = "";
                bool bout = checkTLSClientHello(buf, len, bufSNI);
                if(bout)
                {
                    if(IPport_a_Hash==0 && bSou)
                    {
                        type_HT = 2;
                        bBuf = CPacketTools::getHashBuf_IPport(lppck, bufByte, len, true);
                        if(bBuf)
                           	IPport_a_Hash = CHashTools::HashBuffer(bufByte, len, 32);
                        bBuf = CPacketTools::getHashBuf_IPport(lppck, bufByte, len, false);
                        if(bBuf)
                           	IPport_b_Hash = CHashTools::HashBuffer(bufByte, len, 32);
                    }
                    vctString.push_back(bufSNI);
                }
            }
        }
    }
    return true;
}

bool CFlowHT::intoElephant(CPacket* lppck)
{
    return true;
}

bool CFlowHT::saveObject(FILE* fp, uint64_t cntP, bool bFin)
{
    if(fp && type_HT>0 && cntPck>flow_filter)
    {
        fprintf(fp, "%u,---,---,---\n", cntPck);
        char buf_IPP[UINT8_MAX];
        char buf_type[20];
        CPacketTools::getStr_IPportpair_from_hashbuf(bufKey, lenKey, buf_IPP);
        if(type_HT==1)
            strcpy(buf_type, "GET,");
        else if(type_HT==2)
            strcpy(buf_type, "TLS CH,");

        for(vector<string>::iterator iter=vctString.begin(); iter!=vctString.end(); ++iter)
            fprintf(fp, "%s%u,%u,%u,%s,%s\n", buf_IPP, selfHash, IPport_a_Hash, IPport_b_Hash, buf_type, (*iter).c_str());
        return true;
    }
    else
        return false;
}

//=============================================================================================================
//=============================================================================================================
//=============================================================================================================
//=============================================================================================================
//=============================================================================================================

CFlowHTCreator::CFlowHTCreator(packet_statistics_object_type type, std::string fname, int filter)
{
    pso_type = type;
    strName = fname + ".FHT.csv";
    FILE* fp = fopen(strName.c_str(), "wt");
    if(fp)
        fclose(fp);
    else
        cout << "create file:" << strName << " error!" << endl;
    FHT_filter = filter;
}

IFlow2Object* CFlowHTCreator::create_Object(uint8_t* buf, int len)
{
    CFlowHT* lpFHT = new CFlowHT(buf, len, this, FHT_filter);
    return lpFHT;
}

int CFlowHTCreator::filter_packet(CPacket* lppck)
{
    return 1;
}
