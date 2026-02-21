#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#include <cmath>
#include <math.h>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include "AmrDG.h"
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_Geometry.H>

#include <climits>

void AmrDG::get_u_from_u_w(int c, int i, int j, int k,
                          amrex::Vector<amrex::Array4<amrex::Real>>* uw,
                          amrex::Vector<amrex::Array4<amrex::Real>>* u,
                          const amrex::Vector<amrex::Real>& xi)
{
  //computes the sum of modes and respective basis function evaluated at specified location
  //for all solution components
  for(int q=0 ; q<Q; ++q){
    amrex::Real sum = 0.0;
    for (int n = 0; n < Np_s; ++n){
      sum+=(((*uw)[q])(i,j,k,n)*phi_s(n, xi));
    }
    ((*u)[q])(i,j,k,c) = sum;
  }
}

amrex::Real AmrDG::minmodB(amrex::Real a1,amrex::Real a2,amrex::Real a3,
                          bool &troubled_flag, int l) const
{
  auto _mesh = mesh.lock();
  amrex::Real h;
  auto const dx = _mesh->get_Geom(l).CellSizeArray();
  h = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});

  if(std::abs(a1)<= AMR_TVB_C[l]*TVB_M*std::pow(h,2.0))
  {
    troubled_flag = false;
    return a1;
  }
  else
  {
    troubled_flag = true;
    return minmod(a1,a2,a3, troubled_flag);
  }
}

amrex::Real AmrDG::minmod(amrex::Real a1,amrex::Real a2,amrex::Real a3,
                          bool &troubled_flag) const
{
  bool sameSign = (std::signbit(a1) == std::signbit(a2)) &&
                (std::signbit(a2) == std::signbit(a3));
  int sign;
  if(std::signbit(a1) == std::signbit(-1))
  {sign = -1;}
  else
  {sign = +1;}

  if(sameSign)
  {
    troubled_flag = true;
    return sign*std::min({std::abs(a1), std::abs(a2), std::abs(a3)});
  }
  else
  {
    //slopes disagree: cell is at an extremum, zero the slope
    troubled_flag = true;
    return 0;
  }
}
