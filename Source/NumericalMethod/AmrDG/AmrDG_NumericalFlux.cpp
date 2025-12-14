#include "AmrDG.h"

using namespace amrex;

amrex::Real AmrDG::LLF_numflux(int d, int m,int i, int j, int k, 
                            amrex::Array4<const amrex::Real> up, 
                            amrex::Array4<const amrex::Real> um, 
                            amrex::Array4<const amrex::Real> fp,
                            amrex::Array4<const amrex::Real> fm,  
                            amrex::Array4<const amrex::Real> dfp,
                            amrex::Array4<const amrex::Real> dfm)
{
  
  //implementation of the numerical flux across interface
  //---------------------
  //|         |         |
  //| idx-1 L | R  idx  | 
  //|         |         | 
  //---------------------
  //uL(idx) == up(idx-1)
  //uR(idx) == um(idx)

  amrex::Real C;
  int shift[] = {0,0,0};
  amrex::Real uR,uL,fR,fL,DfR,DfL;

  shift[d] = -1;
  
  //L,R w.r.t boundary plus L==idx, R==idx+1
  uL  = up(i+shift[0],j+shift[1],k+shift[2],m);
  uR  = um(i,j,k,m);   
  fL  = fp(i+shift[0],j+shift[1],k+shift[2],m);
  fR  = fm(i,j,k,m);     
  DfL = dfp(i+shift[0],j+shift[1],k+shift[2],m);
  DfR = dfm(i,j,k,m);     
  C = (amrex::Real)std::max((amrex::Real)std::abs(DfL),(amrex::Real)std::abs(DfR));

  return 0.5*(fL+fR)-0.5*C*(uR-uL);  
}