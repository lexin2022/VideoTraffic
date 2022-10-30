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

#include <iostream>
#include <cstring>
#include "_lib.h/libconfig.h++"
#include "VideoTraffic/video_flows_set.h"

using namespace std;
using namespace libconfig;

int calRatio(int num)
{
    int iout;

    switch (num)
    {
    case 0:
        iout = 1;
        break;
    case 1:
        iout = 8;
        break;
    case 2:
        iout = 16;
        break;
    case 3:
        iout = 32;
        break;
    case 4:
        iout = 64;
        break;
    case 5:
        iout = 128;
        break;
    case 6:
        iout = 256;
        break;
    case 7:
        iout = 512;
        break;
    case 8:
        iout = 1024;
        break;
    case 9:
        iout = 2048;
        break;
    case 10:
        iout = 4096;
        break;
    case 11:
        iout = 8192;
        break;
    case 12:
        iout = 16384;
        break;
    case 13:
        iout = 32768;
        break;
    case 14:
        iout = 65536;
        break;
    default:
        iout = 256;
        break;
    }
    return iout;
}

int main(int argc, char *argv[])
{
    char buf[UINT8_MAX] = "data.cfg";

    if(argc==2)
        strcpy(buf, argv[1]);

    std::cerr << "video flows begin" << std::endl;        

    Config cfg;
    try
    {
        cfg.readFile(buf);
    }
    catch(...)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return(EXIT_FAILURE); 
    }    

    try
    {
        string name = cfg.lookup("VF_File");    
        cout << "video flows pcap file name: " << name << endl;
//        string bigdata_name = cfg.lookup("VF_bigdata_csv");    
//        cout << "big data server name: " << bigdata_name << endl;
                            
        int minpck, stat1, stat2, stat3;
        double dbIdle;

        cfg.lookupValue("VF_Elephant", minpck);
        cout << "VF Elephant Pck. threshold:: " << minpck << endl;
        cfg.lookupValue("VF_Stat1", stat1);
        cout << "Stat. 1 value:" << stat1 << endl;
        cfg.lookupValue("VF_Stat2", stat2);
        cout << "Stat. 2 value:" << stat2 << endl;
        cfg.lookupValue("VF_Stat3", stat3);
        cout << "Stat. 3 value:" << stat3 << endl;
        cfg.lookupValue("VF_Idle", dbIdle);
        cout << "idle time > " << dbIdle << endl;

        int  rno, ratio, seed, beginP=0;
        cfg.lookupValue("VF_ratio", rno);
        ratio = calRatio(rno);
        cout << "sample rate: " << ratio << endl;
        cfg.lookupValue("VF_seed", seed);
        cout << "random seed: " << seed << endl;
        srand(seed);
        if(ratio>1)
            beginP = rand() % ratio;

        packet_statistics_object_type typeS = pso_IPPortPair;
        IFilterStat* lpFS = CFilterStatCreator::create_filter_stat(name, minpck, 26, stat1);
        CVFCreator* lpCreator = new CVFCreator(typeS, name, stat1, stat2, stat3, dbIdle, ratio, seed);
//        bool bBigD = lpCreator->getBigData(bigdata_name);
        if(lpFS && lpCreator)
        {
            lpFS->setParameter(typeS, 3, psm_SouDstDouble, false);
            lpFS->setCreator(lpCreator);
            if(lpFS->isChecked())
            {
//                lpFS->iterPcap();
                lpFS->iterSamplePcap(ratio, beginP);
            }
        }
        else
            cout << "initial error!" << endl;
        if(lpFS)
            delete lpFS;
        if(lpCreator)
            delete lpCreator;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return(EXIT_FAILURE);
    }
    
    return 0;
}