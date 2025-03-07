/*



void AmrDG::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& ba, 
                                    const DistributionMapping& dm)
{ 
  //Print(*ofs) <<"AmrDG::MakeNewLevelFromScratch() "<< lev<<"\n";
  //create a new level from scratch, e.g when regrid criteria for finer level 
  //reached for the first time
  //called when initializing the simulation
  InitData_system(lev,ba,dm);  
  
  if(lev ==0)
  {
    //init both valid and ghost data
    InitialCondition(lev);
  }
  else
  {
    //init valid and ghost data by scattering from coarse

    for(int q=0 ; q<Q; ++q){
      FillCoarsePatch(lev, time, U_w[lev][q], 0, Np,q);
      //for ghost at fine-coarseinterface just copy from coarse
      FillPatchGhostFC(lev,time,q);
    }        
  }  
}


//Remake an existing level using provided BoxArray and DistributionMapping and 
//fill with existing fine and coarse data.
void AmrDG::RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm)
{
  //Print(*ofs) << "RemakeLevel   "<< lev<<"\n";
  
  amrex::Vector<amrex::MultiFab> new_mf;
  new_mf.resize(Q);
  for(int q=0 ; q<Q; ++q){
    new_mf[q].define(ba, dm, Np, nghost);
    new_mf[q].setVal(0.0);    
    FillPatch(lev, time, new_mf[q], 0, Np,q);  
  } 
     
  //clear existing level MFabs defined on old ba,dm
  ClearLevel(lev);
  //create new level MFabs defined on new ba,dm
  InitData_system(lev,ba,dm); 
    
  for(int q=0 ; q<Q; ++q){
    std::swap(U_w[lev][q],new_mf[q]);  
  }
  
  for(int q=0 ; q<Q; ++q){
    FillPatchGhostFC(lev,time, q);
 }
}

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

//Delete level data
void AmrDG::ClearLevel(int lev) 
{
  //Print(*ofs) << "ClearLevel   "<< lev<<"\n";

  //ADER SPECIFIC
  H_w[lev].clear();  
  H[lev].clear();  
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    H_p[lev][d].clear(); 
    H_m[lev][d].clear();
  }
  //SOLVER
  U_w[lev].clear();  
  U[lev].clear();  

  if(model_pde->flag_source_term){S[lev].clear();}
  U_center[lev].clear();  
  
  idc_curl_K[lev].clear();
  idc_div_K[lev].clear();
  idc_grad_K[lev].clear();
  
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    F[lev][d].clear();
    DF[lev][d].clear();
    Fm[lev][d].clear();
    Fp[lev][d].clear();
    DFm[lev][d].clear();
    DFp[lev][d].clear();
    Fnum[lev][d].clear();   
    Fnumm_int[lev][d].clear();
    Fnump_int[lev][d].clear();
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