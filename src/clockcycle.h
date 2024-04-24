/**************************************************************/
/* clock_now(): returns a 64 bit counter of cycles for POWER9 */
/*              clock rate is 512MHz                          */
/**************************************************************/

#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

#include <stdint.h>

uint64_t clock_now(void)
{
  unsigned int tbl, tbu0, tbu1;
  
  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);
  return (((uint64_t)tbu0) << 32) | tbl;
}

#endif // CLOCKCYCLE_H