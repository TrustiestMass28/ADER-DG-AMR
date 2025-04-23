#include "AmrDG.h"

using namespace amrex;

void AmrDG::AMR_clear_level_data(int lev)
{
  //Delete level data

  U_w[lev].clear();  
  U[lev].clear();  
  if(flag_source_term){S[lev].clear();}
  U_center[lev].clear();  

  F[lev].clear();
  Fm[lev].clear();
  Fp[lev].clear();

  DF[lev].clear();
  DFm[lev].clear();
  DFp[lev].clear();

  Fnum[lev].clear();   

  H_w[lev].clear();  
  H[lev].clear();  

  H_p[lev].clear(); 
  H_m[lev].clear();
}

void AmrDG::AMR_remake_level(int lev, amrex::Real time, const amrex::BoxArray& ba,
                            const amrex::DistributionMapping& dm) 
{
  //Remake level based on new geometry. Only the evolved solution vector
  //has to be preserved, i.e only U_w. The data transfer from old MFab
  //to new MFab (differend distribution mapping) of U_w is handled by 
  //FillPatch. All the other vector of MFab are cleared and re-filled
  //with new MFabs based on new geometry and distr.map

  auto _mesh = mesh.lock();

  amrex::Vector<amrex::MultiFab> _mf;
  _mf.resize(Q);
  for(int q=0 ; q<Q; ++q){
    _mf[q].define(ba, dm, basefunc->Np_s, _mesh->nghost);
    _mf[q].setVal(0.0);    
    //FillPatch(lev, time, _mf[q], 0, basefunc->Np_s,q);  //TODO
  } 

  //clear existing level MFabs defined on old ba,dm
  _mesh->ClearLevel(lev);

  //create new level MFabs defined on new ba,dm
  set_init_data_system(lev,ba,dm); 

  //swap old solution MFab with newly created one
  for(int q=0 ; q<Q; ++q){
    std::swap(U_w[lev][q],_mf[q]);  
  }

  //TODO: is step below needed?
  //for(int q=0 ; q<Q; ++q){
  //  FillPatchGhostFC(lev,time, q);
  //}
}

void AmrDG::AMR_make_new_fine_level(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm)
{

}



/*


//Make a new level using provided BoxArray and DistributionMapping and fill with 
//interpolated coarse level data.
void AmrDG::MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, 
  const amrex::DistributionMapping& dm)
{
//Print(*ofs) << "make new level from coarse :   "<< lev<< "\n";
InitData_system(lev,ba,dm); 
for(int q=0 ; q<Q; ++q){
FillCoarsePatch(lev, time, U_w[lev][q], 0, Np,q);
FillPatchGhostFC(lev,time,q);
}
}



//Fillpatch operations fill all cells, valid and ghost, from actual valid data at 
//that level, space-time interpolated data from the next-coarser level, 
//neighboring grids at the same level, and domain boundary conditions 
//(for examples that have non-periodic boundary conditions).
//NB: this function is used for regrid and not for timestepping
void AmrDG::FillPatch(int lev, Real time, amrex::MultiFab& mf,int icomp, int ncomp, int q)
{  
  //Print(*ofs) << "FillPatch   "<< lev<< " |component  "<<q<<"\n";

  if (lev == 0)
  { 
    amrex::Vector<MultiFab*> smf;
    amrex::Vector<Real> stime;
    GetData(lev, q,time, smf, stime);
    
    amrex::CpuBndryFuncFab bcf(nullptr); 
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> physbcf(geom[lev],bc_w[q],bcf);
    
    amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,geom[lev], physbcf, 0);  
    //FillPatchSingleLevel()  :   fills a MultiFab and its ghost region at a single 
    //                            level of refinement. The routine is flexible enough 
    //                            to interpolate in time between two MultiFabs 
    //                            associated with different times
    //                            
    //                            calls also MultiFab::FillBoundary,
    //                            MultiFab::FillDomainBoundary()
    //                            
  }
  else
  { 
    amrex::Vector<MultiFab*> cmf, fmf;
    amrex::Vector<Real> ctime, ftime;
    GetData(lev-1, q,time, cmf, ctime);
    GetData(lev  , q,time, fmf, ftime);

    amrex::Interpolater* mapper = &custom_interp;
    amrex::CpuBndryFuncFab bcf(nullptr);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(geom[lev-1],bc_w[q],bcf);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fine_physbcf(geom[lev],bc_w[q],bcf);
    
    amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,0, icomp, ncomp, 
                              geom[lev-1], geom[lev],coarse_physbcf, 0, fine_physbcf, 
                              0, refRatio(lev-1),mapper, bc_w[q], 0);
    //FillPatchTwoLevels()    :   fills a MultiFab and its ghost region at a single 
    //                            level of refinement, assuming there is an underlying 
    //                           coarse level. This routine is flexible enough to 
    //                            interpolate the coarser level in time first using 
    //                            FillPatchSingleLevel()

  }
}

//fills ghost cells of fine level at fine-coarse interface with respective
//coarse data. Used during timestepping
void AmrDG::FillPatchGhostFC(int lev,amrex::Real time,int q)
{ 
   
  amrex::Vector<MultiFab*> cmf;
  amrex::Vector<Real> ctime;
  GetData(lev-1, q,time, cmf, ctime);
  amrex::CpuBndryFuncFab bcf(nullptr);
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(geom[lev-1],bc_w[q],bcf);

  std::unique_ptr<FillPatcher<MultiFab>> m_fillpatcher;
  auto& fillpatcher = m_fillpatcher;

  fillpatcher = std::make_unique<FillPatcher<MultiFab>>(U_w[lev][q].boxArray(),
                                                        U_w[lev][q].DistributionMap(),
                                                        geom[lev],
                                                        U_w[lev-1][q].boxArray(),
                                                        U_w[lev-1][q].DistributionMap(),
                                                        geom[lev-1],
                                                        IntVect(nghost),
                                                        Np, 
                                                        //&custom_interp);
                                                        &pc_interp);

  fillpatcher->fillCoarseFineBoundary(U_w[lev][q],IntVect(nghost),time,cmf,ctime,
                                      0,0,Np,coarse_physbcf,0,bc_w[q],0);                        
}	    
	    
// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
//also fills ghost cells
void AmrDG::FillCoarsePatch (int lev, Real time, amrex::MultiFab& mf, 
                            int icomp,int ncomp, int q)
{                               
  amrex::Vector<MultiFab*> cmf;
  amrex::Vector<Real> ctime;
  GetData(lev-1,q, time, cmf, ctime);
  
  amrex::Interpolater* mapper = &custom_interp;
  amrex::CpuBndryFuncFab bcf(nullptr);
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(geom[lev-1],bc_w[q],bcf);
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fine_physbcf(geom[lev],bc_w[q],bcf);
  
  amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], 
                              geom[lev], coarse_physbcf, 0, fine_physbcf, 0, 
                              refRatio(lev-1),mapper, bc_w[q], 0);                               
}

void AmrDG::GetData (int lev, int q, Real time, Vector<MultiFab*>& data, 
                    Vector<Real>& datatime)
{
  data.clear();
  datatime.clear();
  data.push_back(&(U_w[lev][q]));
  datatime.push_back(time);
}

//averages cell centered data from finer cells to the respective covered coarse cell
void AmrDG::AverageFineToCoarse()
{  
  //Print(*ofs) << "AverageFineToCoarse()"<< "\n";
  
  for (int l = finest_level; l > 0; --l){  
    for(int q=0; q<Q; ++q){   
      custom_interp.average_down(U_w[l][q], U_w[l-1][q],0,U_w[l-1][q].nComp(), 
                                refRatio(l-1), l,l-1);
    }
  } 
}

//averages face centered data from finer cells to the respective covered coarse cell
void AmrDG::AverageFineToCoarseFlux(int lev)
{
  if(lev!=finest_level)
  { 
    for(int d = 0; d<AMREX_SPACEDIM; ++d){
      for(int q=0; q<Q; ++q){           
        custom_interp.average_down_flux(Fnumm_int[lev+1][d][q], Fnumm_int[lev][d][q],0,
                                  Fnumm_int[lev][d][q].nComp(), refRatio(lev), 
                                  lev+1,lev,d,true);
        custom_interp.average_down_flux(Fnump_int[lev+1][d][q], Fnump_int[lev][d][q],0,
                                  Fnump_int[lev][d][q].nComp(), refRatio(lev), 
                                  lev+1,lev,d,true);            
      }
    } 
  }
}
*/
/*
void AmrDG::AMR_settings_tune()
{
  /////////////////////////
  //AMR MESH PARAMETERS (tune only if needed)
  //please refer to AMReX_AmrMesh.H for all functions for setting the parameters
  //Set the same blocking factor for all levels
  SetBlockingFactor(2); 
  SetGridEff(0.9);
  //Different blocking factor for each refinemetn level

  //amrex::Vector<int> block_fct;// (max_level+1);
  //for (int l = 0; l <= max_level; ++l) {
  //  if(l==0){block_fct.push_back(8);}
  //  else if(l==1){block_fct.push_back(4);}
  //}
  ////NB: can also specify different block factor per dimension and different
  ////block factor per dimension per level
  //SetBlockingFactor(block_fct);

  
  //SetMaxGridSize(16);
  //iterate_on_new_grids = false;//will genrete only one new level per refinement step
  /////////////////////////
}


*/