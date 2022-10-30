/**
MIT License

Copyright (c) 2021 hwu(hwu@seu.edu.cn), caymanhu(caymanhu@qq.com)

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

/**
 * @file libFilterSE.h
 * @author hwu(hwu@seu.edu.cn), caymanhu(caymanhu@qq.com)
 * @brief filter, 分组过滤器
 * @version 0.1
 * @date 2021-09-14
 */

#ifndef LIB_THRESHOLD_FILTER_SE_H
#define LIB_THRESHOLD_FILTER_SE_H

#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <string>
#include <vector>
#include "libPcapSE.h"
#include "libBaseSE.h"

class IFilterObject
{
public:
    virtual ~IFilterObject() {}
public:
    virtual bool checkObject() = 0;
    virtual bool addPacket(CPacket* lppck, bool bSou) = 0;
    virtual bool intoElephant(CPacket* lppck) = 0;
    virtual bool saveObject(FILE* fp, uint64_t cnt) = 0;

    /**
     * @brief 判断packet是否属于本统计
     * 
     * @param buf 内存指针
     * @param len 比较长度
     * @return true 
     * @return false 
     */
    virtual bool isSameObject(uint8_t* buf, int len) = 0;
public:
    virtual uint32_t getPckCnt() = 0;
    virtual void incPckCnt() = 0;
};

class IFilterObjectCreator
{
public:
    virtual ~IFilterObjectCreator() {}
public:
    virtual IFilterObject* create_Object(uint8_t* buf, int len) = 0;
    //virtual void saveCSVTitle(FILE* fp) = 0;
public:    
    virtual std::string getCSVFname() = 0;
    virtual packet_statistics_object_type getType() = 0;
    virtual bool isSave() = 0;
};

/**
 * @brief 相同条件统计的接口，通过该接口调用里面的函数
 * 
 */
class IFilterStat
{
public:
    virtual ~IFilterStat(){}
public:
    virtual bool iterPcap() = 0;
    virtual bool iterSamplePcap(int ratio, int beginP) = 0;
    virtual bool dealPacket(CPacket* lppck) = 0;
    virtual bool saveData(uint64_t cntPck) = 0;
    virtual std::vector<IFilterObject*>* getElephant() = 0;
public:
    virtual bool isChecked() = 0;
    virtual void setParameter(packet_statistics_object_type type, int protocol, packet_statistics_method method, bool bp) = 0;
    virtual void setCreator(IFilterObjectCreator* lpFIC) = 0;
};

/**
 * @brief 统计工具类
 */
class CFilterStatCreator
{
public:
    static IFilterStat* create_filter_stat(std::string fname, int min_pck, int bit, int min_stat);
};
#endif
