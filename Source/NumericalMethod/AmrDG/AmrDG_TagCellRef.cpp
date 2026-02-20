#include "AmrDG.h"

using namespace amrex;

void AmrDG::AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
  if(!m_tag_impl) return;

  const int tagval = TagBox::SET;

  auto _mesh = mesh.lock();

  const amrex::Real amr_c_lev = _mesh->amr_c[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  {
    #ifdef AMREX_USE_OMP
    for (MFIter mfi(tags, MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(tags, true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.tilebox();
      const auto tagfab = tags.array(mfi);

      amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
      {
        if(m_tag_impl(lev, i, j, k, time, amr_c_lev)){
          tagfab(i,j,k) = tagval;
        }
      });
    }
  }
}
