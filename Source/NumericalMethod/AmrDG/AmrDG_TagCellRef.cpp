#include "AmrDG.h"

using namespace amrex;

void AmrDG::AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
  if(!m_tag_impl) return;

  const int tagval = TagBox::SET;

  auto _mesh = mesh.lock();

  const amrex::Real amr_c_lev = _mesh->amr_c[lev];

  //sync ghost cells so TVB stencil (if used) reads up-to-date neighbor data
  for(int q=0; q<Q; ++q){
    U_w(lev,q).FillBoundary(_mesh->get_Geom(lev).periodicity());
  }

  amrex::Vector<amrex::MultiFab*> state_uw(Q);
  for(int q=0; q<Q; ++q){
    state_uw[q] = &(U_w(lev,q));
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  {
    amrex::Vector<amrex::Array4<amrex::Real>> uw(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(tags, MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(tags, true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.tilebox();
      const auto tagfab = tags.array(mfi);

      for(int q=0; q<Q; ++q){
        uw[q] = state_uw[q]->array(mfi);
      }

      amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
      {
        if(m_tag_impl(lev, i, j, k, time, amr_c_lev, &uw)){
          tagfab(i,j,k) = tagval;
        }
      });
    }
  }
}
