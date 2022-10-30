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
#include "VideoTraffic/flow_HT.h"
#include "winlin/winlinux.h"

using namespace std;
using namespace libconfig;

int main(int argc, char *argv[])
{
    char buf[UINT8_MAX] = "data.cfg";

    if(argc==2)
        strcpy(buf, argv[1]);

    std::cerr << "flow HT begin" << std::endl;        

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
        string name = cfg.lookup("FHT_path");    
        cout << "Flow HT path: " << name << endl;

        int bit, filter;                
        cfg.lookupValue("FHT_bit", bit);
        cout << "statistics Pck. bit threshold:" << bit << endl;
        cfg.lookupValue("FHT_filter", filter);
        cout << "packet filter:" << filter << endl;

        packet_statistics_object_type typeS = pso_IPPortPair;

        if(name.length()>0)
        {
            vector<string> vctFN;
            if(iterPathPcaps(name, &vctFN))
            {
                for(vector<string>::iterator iter=vctFN.begin(); iter!=vctFN.end(); ++iter)
                {
                    string strFN = *iter;
                    cout << "pcap file:" << strFN << endl;
                    IFlow2Stat* lpFS = CFlow2StatCreator::create_flow2_stat(strFN, bit, 1);
                    IFlow2ObjectCreator* lpFOC = new CFlowHTCreator(typeS, strFN, filter);
                    if(lpFS && lpFOC)
                    {
                        lpFS->setParameter(typeS, 1, psm_SouDstDouble, true);
                        lpFS->setCreator(lpFOC);
                        if(lpFS->isChecked())
                        {
                            lpFS->iterPcap();
                        }
                    }
                    else
                        cout << "pcap file " << strFN << " open error!" << endl;
                    if(lpFS)
                        delete lpFS;
                    if(lpFOC)
                        delete lpFOC;
                }
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return(EXIT_FAILURE);
    }
    
    return 0;
}