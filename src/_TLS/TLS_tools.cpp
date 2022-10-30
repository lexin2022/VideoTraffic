/*
 * @Author: your name
 * @Date: 2021-10-19 20:09:36
 * @LastEditTime: 2021-11-06 17:01:54
 * @LastEditors: your name
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \conf2022g:\Code\gitee\conf2022\src\_TLS\TLS_tools.cpp
 */
#include "_TLS/TLS_tools.h"

int getParaBeforeExtensions(uint8_t *buffer, int &lenCH)
{
    int iout = -1;
    int pos, lenPara, newpos;

    if(lenCH>43)                                                //tls header 5 + CH header 6 + GMT time 4 + random byte 28
    {
        pos = 43;
        lenPara = buffer[pos];                                  // session id
        newpos = pos+1+lenPara;
        if(lenPara>=0 && lenCH>newpos)
        {
            pos = newpos;
            lenPara = buffer[pos]*256+buffer[pos+1];            //cipher suites
            newpos = pos+2+lenPara;
            if(lenPara>0 && lenCH>newpos)
            {
                pos = newpos;
                lenPara = buffer[pos];                          //compression method
                newpos = pos+1+lenPara;
                if(lenPara>0 && lenCH>newpos)
                {
                    /*
                    if(buffer[pos+1]==0)
                        iCompression = 0;
                    else
                        iCompression = 1;                        
                    */
                    pos = newpos;
                    lenPara = buffer[pos]*256+buffer[pos+1];    //extensions
                    if(pos+lenPara < lenCH)
                        lenCH = pos+lenPara;
                    iout = pos+2;                               //extensions detail            
                }
            }
        }
    }
    return iout;
}

bool getSName_from_CH_extensions(unsigned char *buffer, int lenCH, int posB, char *bufSNI)
{
    bool bout = false;
    int pos = posB;
    int lenExt;

    while(pos < lenCH)
    {
        if(buffer[pos]==0 && buffer[pos+1]==0){  //server
            //server_name 2 + len 2 + sn list len 2 + host_name(0) 1
            if(buffer[pos+6]==0){
                lenExt = buffer[pos+7]*256+buffer[pos+8];
                if(lenExt<100){
                    memcpy(bufSNI, buffer+pos+9, lenExt);
                    bufSNI[lenExt] = 0;
                    bout = true;
                }
            }
            break;
        }
        lenExt = 4 + buffer[pos+2]*256+buffer[pos+3]; // type + lenbuf + data
        pos += lenExt;
    }

    return bout;
}

bool getServerHostName(uint8_t *buffer, int len, char *bufSNI)
{
    bool bout = false;
    int lenCH, pos;

    lenCH = buffer[3]*256+buffer[4]+5;
    if(lenCH > len )
        lenCH = len;
    pos = getParaBeforeExtensions(buffer, lenCH);
    if(pos > 0)
    {
        bout = getSName_from_CH_extensions(buffer, lenCH, pos, bufSNI);
    }
    return bout;
}


bool checkTLSClientHello(uint8_t *buffer, int len, char *bufSNI)
{
    bool bOut = false;
    if(len>100 && buffer[0]==0x16 && buffer[1]==3 && buffer[5]==1){
        bOut = getServerHostName(buffer, len, bufSNI);
    }
    return bOut;
}