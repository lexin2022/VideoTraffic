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

#include "VideoTraffic/video_flows_set.h"
#include "_lib.h/libHashSE.h"
#include "_lib.h/libPcapSE.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

CVideoFlows::CVideoFlows(uint8_t* buf, int len, IFilterObjectCreator* lpFOC, int num1, int num2, int num3, double tmIdle)
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

    stat1 = num1;
    stat2 = num2;
    stat3 = num3;
    time_idle = tmIdle;
    initial();
}

CVideoFlows::~CVideoFlows()
{
    if(bufKey)
        free(bufKey);
}

bool CVideoFlows::checkObject()
{
    if(lenKey>0 && bufKey)
        return true;
    else
        return false;
}

bool CVideoFlows::isSameObject(uint8_t* buf, int len)
{
    bool bout = false;

    if(lenKey == len)
    {
        if(memcmp(bufKey, buf, len)==0)
            bout = true;
    }
    return bout;
}

bool CVideoFlows::addPacket(CPacket* lppck, bool bSou)
{
    if(lppck)
    {
        int lenpl, area;
        if(bSou)
        {
            o_pck ++;
            o_len += lppck->getLenPayload();

            lenpl = lppck->getLenPayload();
            if(lenpl == 0) 
                cnt_o_area_pck[0]++;
            else
            {
                o_dp ++;
                if(lenpl>=1300) 
                    cnt_o_upperlimit_pck += calLimit(lenpl);
                else
                {
                    area = lenpl/100 + 1;
                    if(area>0 && area<=13)
                        cnt_o_area_pck[area] ++;
                }
            }

            //label
            if(iLabel < 0)
            {
                CVFCreator* lpVFC = dynamic_cast<CVFCreator*>(lpCreator);
                if(lppck->getIPVer()==4)
                    iLabel = lpVFC->checkBigDataLabel(lppck->getProtocol(), lppck->getSrcIP4(), lppck->getSrcPort());
                else
                    iLabel = 0;
                
                if(iLabel>0)
                    bPulse = true;
            }
        }
        else
        {
            i_pck ++;
            i_len += lppck->getLenPayload();

            lenpl = lppck->getLenPayload();
            if(lenpl == 0) 
                cnt_i_area_pck[0]++;
            else
            {
                i_dp ++;
                if(lenpl>=1300) 
                    cnt_i_upperlimit_pck += calLimit(lenpl);
                else
                {
                    area = lenpl/100 + 1;
                    if(area>0 && area<=13)
                        cnt_i_area_pck[area] ++;
                }
            }
        }
        if(beginTM<0)
            beginTM = lppck->getPckOffTime();
        if(endTM>=0 && lppck->getPckOffTime()-endTM>time_idle)
            freeTM += lppck->getPckOffTime()-endTM;
        if(bPulse)
        {
            if(tmPulseBegin<0)
                tmPulseBegin = lppck->getPckOffTime();
            else
            {
                if(lppck->getPckOffTime()-endTM>time_idle)
                {
                    st_pulse stPulse;
                    stPulse.time = tmPulseBegin;
                    stPulse.tmData = endTM - tmPulseBegin;
                    if(stPulse.tmData<0.001)
                        stPulse.tmData = 0.001;
                    stPulse.tmIdle = lppck->getPckOffTime()-endTM;
                    vctPulse.push_back(stPulse);
                    tmPulseBegin = lppck->getPckOffTime();
                }
            }
        }
        endTM = lppck->getPckOffTime();

        if(cntPck == stat1)
            saveStat(stat1, 1);
        if(cntPck == stat2)
            saveStat(stat2, 2);
        if(cntPck == stat3)
            saveStat(stat3, 3);

        return true;
    }
    else
        return false;

}

bool CVideoFlows::intoElephant(CPacket* lppck)
{
    return true;
}

bool CVideoFlows::saveObject(FILE* fp, uint64_t cnt)
{
    if(fp)
        fclose(fp);

    if(bPulse)
    {
        string strFN = lpCreator->getCSVFname() + ".pulse.csv";
        FILE* lpF = fopen(strFN.c_str(), "at");
        if(lpF)
        {
            char bufMsg[UINT8_MAX];
            CPacketTools::getStr_IPportpair_from_hashbuf(bufKey, lenKey, bufMsg);

            for(vector<st_pulse>::iterator iter=vctPulse.begin(); iter!=vctPulse.end(); ++iter)
            {
                fprintf(lpF, "%s%d,%.4f,%.4f,%.4f\n", bufMsg, iLabel, 
                        (*iter).time, (*iter).tmData, (*iter).tmIdle);
            }
            fclose(lpF);
        }
    }
    return true;
}

void CVideoFlows::initial()
{
    beginTM = endTM = -1;
    freeTM = 0;
    i_pck = o_pck = i_len = o_len = i_dp = o_dp = 0;
    for(int i=0; i<14; i++)
        cnt_o_area_pck[i] = cnt_i_area_pck[i] = 0;
    cnt_i_upperlimit_pck = cnt_o_upperlimit_pck = 0;
    bPulse = false;
    tmPulseBegin = -1;
    iLabel = -1;
}

double CVideoFlows::calLimit(int len)
{
    double dbout;

    dbout = len;
    dbout /= 1300;

    return dbout;
}

bool CVideoFlows::saveStat(int statnum, int step)
{
    bool bout = false;

    string str = lpCreator->getCSVFname();
    string strName = str + to_string(statnum) + ".csv";

    FILE* fp = fopen(strName.c_str(), "at");
    if(fp)
    {
        char bufMsg[UINT8_MAX];
        CPacketTools::getStr_IPportpair_from_hashbuf(bufKey, lenKey, bufMsg);

        double spd_o_pck=0, spd_i_pck=0, spd_o_len=0, spd_i_len=0;
        double r_o_pck, r_o_data, r_o_len;
        double tmGap = endTM-beginTM-freeTM;
        if(tmGap<=0)
            tmGap = 10000;
        r_o_pck = (double)o_pck/(double)(o_pck+i_pck);
        r_o_data = (double)o_dp/(double)(o_dp+i_dp);
        r_o_len = (double)o_len/(double)(o_len+i_len);
        if(tmGap>0)
        {
            spd_o_pck = o_pck/tmGap;
            spd_i_pck = i_pck/tmGap;
            spd_o_len = o_len/tmGap;
            spd_i_len = i_len/tmGap;
        }

        int il = 0;
        if(iLabel>0)
            il = iLabel;
        fprintf(fp, "%s%u,%d,%u,%u,%.6f,%u,%u,%.6f,%lu,%lu,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,",
                bufMsg, selfHash, il,
                o_pck, i_pck, r_o_pck,
                o_dp, i_dp, r_o_data,
                o_len, i_len, r_o_len,
                beginTM, endTM,
                spd_o_pck, spd_i_pck, spd_o_len, spd_i_len
                );
        for(int i=0; i<14; i++)
            fprintf(fp, "%u,", cnt_o_area_pck[i]);
        fprintf(fp, "%.0f,", cnt_o_upperlimit_pck);

        for(int i=0; i<14; i++)
            fprintf(fp, "%u,", cnt_i_area_pck[i]);
        fprintf(fp, "%.0f,", cnt_i_upperlimit_pck);

        fprintf(fp, "\n");
        fclose(fp);
        bout = true;
    }

    return bout;
}

//===============================================================================================================
//===============================================================================================================
//===============================================================================================================
//===============================================================================================================

CVFCreator::CVFCreator(packet_statistics_object_type type, string fname, int num1, int num2, int num3, double tmIdle, int ratio, int seed)
{
    pso_type = type;
    strName = fname;
    statnum1 = num1;
    statnum2 = num2;
    statnum3 = num3;
    tm_idle = tmIdle;

    strName = fname + ".flow.stat." +  "r_" + to_string(ratio)  + ".s_" + to_string(seed) + ".";
    
    string strFN;
    strFN = strName + to_string(statnum1) + ".csv";
    saveTitle(strFN.c_str(), 1);
    strFN = strName + to_string(statnum2) + ".csv";
    saveTitle(strFN.c_str(), 2);
    strFN = strName + to_string(statnum3) + ".csv";
    saveTitle(strFN.c_str(), 3);

    strFN = strName + ".pulse.csv";
    FILE* lpF = fopen(strFN.c_str(), "wt");
    if(lpF)
    {
        fprintf(lpF, "protocol,s_IP,s_port,d_IP,d_port,label,begin_time,T1,T0\n");
        fclose(lpF);
    }
    else
        cout << "open file:" << strFN << " error!" <<endl;
}

void CVFCreator::saveTitle(const char* name, int stat)
{
    FILE* fp = fopen(name, "wt");
    if(fp)
    {
        fprintf(fp, "protocol,s_IP,s_port,d_IP,d_port,hash,label,o_pck,i_pck,r_o_pck,o_data_p,i_data_p,r_o_dp,o_len,i_len,r_o_len,begin TM,end TM,o_spd_pck,i_spd_pck,o_spd_len,i_spd_len,"); //lbegin TM,end TM,cntStat,");
        for(int i=0; i<14; i++)
        {
            if(i==0)
                fprintf(fp, "o_(%d),", i);
            else
                fprintf(fp, "o_(%d-%d),", (i-1)*100+1, i*100);
        }
        fprintf(fp, "o_(>1300),");
        
        for(int i=0; i<14; i++)
        {
            if(i==0)
                fprintf(fp, "i_(%d),", i);
            else
                fprintf(fp, "i_(%d-%d),", (i-1)*100+1, i*100);
        }
        fprintf(fp, "i_(>1300),");

        fprintf(fp, "\n");
        fclose(fp);
    }
    else
        cout << "open file:" << name << " error!" <<endl;
}

IFilterObject* CVFCreator::create_Object(uint8_t* buf, int len)
{
    IFilterObject* lpFI = new CVideoFlows(buf, len, this, statnum1, statnum2, statnum3, tm_idle);
    return lpFI;
}
    
bool CVFCreator::getBigData(string strFN)
{
    st_big_data stBigData;
    ifstream fin(strFN.c_str());
    const int LINE_LENGTH = 4096 ;
    char str[LINE_LENGTH];  
    int line = 0;
    while ( fin.getline(str, LINE_LENGTH) )
    {    
        if(line==0)
        {
            if(strcmp(str,"protocol,s_IP,s_port,label")!=0)
            {
                cout << "big data csv error!" << endl;
                break;
            }
        }
        else
        {
            char *lpbuf, *lppos;
            int len = strlen(str);
            lppos = strstr(str, ",");
            if(lppos)
            {
                lpbuf = lppos + 1;
                lppos[0] = 0;
                stBigData.protocol = (uint8_t)atoi(str);
                lppos = strstr(lpbuf, ",");
                if(lppos)
                {
                    lppos[0] = 0;
                    stBigData.IP = CPacketTools::trans2IP(lpbuf);
                    lpbuf = lppos + 1;
                    lppos = strstr(lpbuf, ",");
                    if(lppos)
                    {
                        lppos[0] = 0;
                        stBigData.port = (uint16_t)atoi(lpbuf);
                        lpbuf = lppos + 1;
                        stBigData.label = (uint8_t)atoi(lpbuf);
                        if(stBigData.IP>0 && stBigData.protocol>0 && stBigData.port>0 && stBigData.label>0)
                            vctBigData.push_back(stBigData);
                    }
                }
            }
        }
        line++;
    }   

    cout << "load big data " << vctBigData.size() << " services" << endl;
    return true;
}

int CVFCreator::checkBigDataLabel(uint8_t prot, uint32_t IP, uint16_t port)
{
    int iout = 0;

    for(vector<st_big_data>::iterator iter=vctBigData.begin(); iter!=vctBigData.end(); ++iter)
    {
        if((*iter).protocol == prot &&
            (*iter).IP == IP &&
            (*iter).port == port)
        {
            iout = (*iter).label;
            break;
        }
    }

    return iout;
}
