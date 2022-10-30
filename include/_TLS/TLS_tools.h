#ifndef TLS_TOOLS_H
#define TLS_TOOLS_H

#include <stdint.h>
#include <cstring>

bool checkTLSClientHello(uint8_t *buffer, int len, char *bufSNI);

#endif