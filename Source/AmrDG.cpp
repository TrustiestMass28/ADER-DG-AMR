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
//#include "Compressible_Euler.h"
//#include "Advection.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

using namespace amrex;

//AmrDG class constructor, also initializes the AmrCore class (used for AMR)
AmrDG::AmrDG(Simulation* _sim,const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
              int _coord, Vector<IntVect> const& _ref_ratios,  
              Array<int,AMREX_SPACEDIM> const& _is_per, 
              amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo, 
              amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi, 
              amrex::Vector<amrex::Vector<int>> _bc_lo_type, 
              amrex::Vector<amrex::Vector<int>> _bc_hi_type,amrex::Real _T,
              amrex::Real _CFL, int _p,
              int _t_regrid, int _t_outplt, std::string _limiter_type, 
              amrex::Real _TVB_M,
              amrex::Vector<amrex::Real> _AMR_TVB_C ,
              amrex::Vector<amrex::Real> _AMR_curl_C, 
              amrex::Vector<amrex::Real> _AMR_div_C,  
              amrex::Vector<amrex::Real> _AMR_grad_C, 
              amrex::Vector<amrex::Real> _AMR_sec_der_C,
              amrex::Real _AMR_sec_der_indicator, 
              amrex::Vector<amrex::Real> _AMR_C, int _t_limit) 
              
              :  AmrCore (_rb, _max_level, _n_cell, _coord, _ref_ratios, _is_per)
{ 
  sim =_sim;
  
  //I/O settings
  t_outplt = _t_outplt;
  
  //AMR SETTINGS 
  L = _max_level+1;
  t_regrid = _t_regrid;
  t_limit  = _t_limit;
  AMR_curl_C = _AMR_curl_C;
  AMR_div_C = _AMR_div_C;
  AMR_grad_C = _AMR_grad_C;
  AMR_sec_der_C = _AMR_sec_der_C;
  AMR_sec_der_indicator = _AMR_sec_der_indicator;
  AMR_TVB_C = _AMR_TVB_C;
  AMR_C = _AMR_C;
  
  //DG SETTINGS
  CFL = _CFL;
  p = _p; 
  T = _T;
  limiter_type = _limiter_type;
  TVB_M = _TVB_M;
  
  Q = sim->model_pde->Q_model;  
  Q_unique = sim->model_pde->Q_model_unique;
  
  Print() <<"Solving system with:"<<"\n";
  Print() <<"   total equations   "<<Q<<"\n";
  Print() <<"   unique equations  "<<Q_unique<<"\n";
  
  //number of modes and nodes
  number_modes();
  number_quadintpts();

  Print() <<"Np         : "<<Np<<"\n";
  Print() <<"mNp        : "<<mNp<<"\n";
  Print() <<"qMp        : "<<qMp<<"\n";
  Print() <<"qMpbd      : "<<qMpbd<<"\n";
  Print() <<"qMp_L2proj : "<<qMp_L2proj<<"\n";
  /*
  Np    : number of modes
  mNp   : number of modes for modified basis function
  mMp   : number of interpolation nodes (equidistant) related to mNp
  qMp   : number of quadrature  points (Gauss-Lobatto distribution) 
  mMpbd  : number of interpolation nodes on a boundary (equidistant) related to mNp
  qMpbd : number of quadrature points on a boundary (Gauss-Lobatto distribution)
  */

  U_w.resize(L); 
  H_w.resize(L); 
  H.resize(L); 
  U.resize(L); 
  if(sim->model_pde->flag_source_term){S.resize(L);}
  U_center.resize(L); 
  F.resize(L);
  Fm.resize(L);
  Fp.resize(L);
  DF.resize(L);
  DFm.resize(L);
  DFp.resize(L);
  H_p.resize(L);
  H_m.resize(L);
  Fnum.resize(L);
  Fnumm_int.resize(L);
  Fnump_int.resize(L);
  
  idc_curl_K.resize(L);
  idc_div_K.resize(L);
  idc_grad_K.resize(L);
  
  //basis functions d.o.f mapper
  mat_idx_s.resize(Np, amrex::Vector<int>(AMREX_SPACEDIM));
  mat_idx_st.resize(mNp, amrex::Vector<int>(AMREX_SPACEDIM+1));  
  PhiIdxGenerator_s();
  PhiIdxGenerator_st();

  //Gaussian quadrature
  xi_ref_GLquad_s.resize( (int)std::pow(qMp_1d,AMREX_SPACEDIM),
                  amrex::Vector<amrex::Real> (AMREX_SPACEDIM));
                  
  xi_ref_GLquad_t.resize(qMp_1d,amrex::Vector<amrex::Real> (1)); 
  xi_ref_equidistant.resize(qMp,amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1));  
  xi_ref_GLquad.resize(qMp,amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1));  
  xi_ref_GLquad_L2proj.resize(qMp_L2proj,amrex::Vector<amrex::Real> (AMREX_SPACEDIM)); 
  xi_ref_GLquad_bdm.resize(AMREX_SPACEDIM,
                    amrex::Vector<amrex::Vector<amrex::Real>> (qMpbd,
                    amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)));                    
  xi_ref_GLquad_bdp.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Vector<amrex::Real>> (qMpbd,
                    amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)));
  volquadmat.resize(Np,amrex::Vector<amrex::Real>(qMp));  
  L2proj_quadmat.resize(Np,amrex::Vector<amrex::Real>(qMp_L2proj));  
  GenerateQuadPts();

  //Initialize generalized Vandermonde matrix (only volume, no boudnary version)
  //and their inverse
  V.resize(qMp,amrex::Vector<amrex::Real> (mNp)); 
  Vinv.resize(mNp,amrex::Vector<amrex::Real> (qMp));
  VandermondeMat();
  InvVandermondeMat();
  
  //Element matrices for ADER-DG corrector
  Mk_corr.resize(Np,amrex::Vector<amrex::Real>(Np));
  Sk_corr.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Vector<amrex::Real>>(Np,
          amrex::Vector<amrex::Real>(qMp)));
  Mkbd.resize((int)(2*AMREX_SPACEDIM), amrex::Vector<amrex::Vector<amrex::Real>>(Np,
          amrex::Vector<amrex::Real>(qMpbd)));
  
  //Element matrices for predictor evolution
  Mk_h_w.resize(mNp,amrex::Vector<amrex::Real>(mNp));
  Mk_h_w_inv.resize(mNp,amrex::Vector<amrex::Real>(mNp));
  Mk_pred.resize(mNp,amrex::Vector<amrex::Real>(Np));  
  Sk_pred.resize(AMREX_SPACEDIM, amrex::Vector<amrex::Vector<amrex::Real>>(mNp,
            amrex::Vector<amrex::Real>(mNp)));
  Mk_s.resize(mNp,amrex::Vector<amrex::Real>(mNp));
  Sk_predVinv.resize(AMREX_SPACEDIM, amrex::Vector<amrex::Vector<amrex::Real>>(mNp,
            amrex::Vector<amrex::Real>(qMp)));
  Mk_sVinv.resize(mNp,amrex::Vector<amrex::Real>(qMp));
  MatrixGenerator();
  
  //Amrex boundary data
  bc_w.resize(Q,amrex::Vector<amrex::BCRec>(Np));
  for(int q=0; q<Q; ++q){
    for (int n = 0; n < Np; ++n){
      for (int d = 0; d < AMREX_SPACEDIM; ++d)
      {   
        bc_w[q][n].setLo(d,_bc_lo[q][d]);
        bc_w[q][n].setHi(d,_bc_hi[q][d]);
      }
    }
  }

  //AmrDG boundary data
  bc_lo_type.resize(Q);
  bc_hi_type.resize(Q);
  
  gDbc_lo.resize(Q);
  gDbc_hi.resize(Q);
  
  gNbc_lo.resize(Q);
  gNbc_hi.resize(Q);
  
  for(int q=0; q<Q; ++q){
    bc_lo_type[q].resize(AMREX_SPACEDIM);
    bc_hi_type[q].resize(AMREX_SPACEDIM);
    
    gDbc_lo[q].resize(AMREX_SPACEDIM);
    gDbc_hi[q].resize(AMREX_SPACEDIM); 
    
    gNbc_lo[q].resize(AMREX_SPACEDIM);
    gNbc_hi[q].resize(AMREX_SPACEDIM);  
    
    for(int d=0; d<AMREX_SPACEDIM; ++d){
        bc_lo_type[q][d] = _bc_lo_type[q][d];
        bc_hi_type[q][d] = _bc_hi_type[q][d];
        
        gDbc_lo[q][d] =gDirichlet_bc(d,-1,q);
        gDbc_hi[q][d] =gDirichlet_bc(d,1,q);
        
        gNbc_lo[q][d] =gNeumann_bc(d,-1,q);
        gNbc_hi[q][d] =gNeumann_bc(d,1,q);             
    }
  }
  
  //Interpolation coarse<->fine data scatter/gather
  custom_interp.getouterref(this); 
  custom_interp.interp_proj_mat();
  
  //Refinement parameters fine tuning
  AMR_settings_tune();
}

AmrDG::~AmrDG() {}

void AmrDG::AMR_settings_tune()
{
  /////////////////////////
  //AMR MESH PARAMETERS (tune only if needed)
  //please refer to AMReX_AmrMesh.H for all functions for setting the parameters
  //Set the same blocking factor for all levels
  SetBlockingFactor(2); 
  SetGridEff(0.9);
  //Different blocking factor for each refinemetn level
  /*
  amrex::Vector<int> block_fct;// (max_level+1);
  for (int l = 0; l <= max_level; ++l) {
    if(l==0){block_fct.push_back(8);}
    else if(l==1){block_fct.push_back(4);}
  }
  //NB: can also specify different block factor per dimension and different
  //block factor per dimension per level
  SetBlockingFactor(block_fct);
  */
  
  //SetMaxGridSize(16);
  //iterate_on_new_grids = false;//will genrete only one new level per refinement step
  /////////////////////////
}
void AmrDG::Init()
{
  //initialize multilevel mesh, geometry, Box array and DistributionMap
  Print() <<"AmrDG::Init()"<<"\n";  
  const Real time = 0.0;
  InitFromScratch(time);
}

void AmrDG::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& ba, 
                                    const DistributionMapping& dm)
{ 
  AllPrint() <<"AmrDG::MakeNewLevelFromScratch() "<< lev<<"\n";
  //create a new level from scratch, e.g when regrid criteria for finer level 
  //reached for the first time
  //called when initializing the simulation
  InitData_system(lev,ba,dm);
  InitialCondition(lev);  
}

void AmrDG::InitData_system(int lev,const BoxArray& ba, const DistributionMapping& dm)
{
  AllPrint() <<"AmrDG::InitData_system() "<< lev<<"\n";
  //init data structures for level for all solution components of the system
  U_w[lev].resize(Q); 
  H_w[lev].resize(Q); 
  U[lev].resize(Q);
  H[lev].resize(Q);   
  if(sim->model_pde->flag_source_term){S[lev].resize(Q);}
  U_center[lev].resize(Q); 
  
  F[lev].resize(AMREX_SPACEDIM);
  Fm[lev].resize(AMREX_SPACEDIM);
  Fp[lev].resize(AMREX_SPACEDIM);
  DF[lev].resize(AMREX_SPACEDIM);
  DFm[lev].resize(AMREX_SPACEDIM);
  DFp[lev].resize(AMREX_SPACEDIM);
  H_p[lev].resize(AMREX_SPACEDIM);
  H_m[lev].resize(AMREX_SPACEDIM);
  Fnum[lev].resize(AMREX_SPACEDIM);
  Fnumm_int[lev].resize(AMREX_SPACEDIM);
  Fnump_int[lev].resize(AMREX_SPACEDIM);
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    F[lev][d].resize(Q);
    Fm[lev][d].resize(Q);
    Fp[lev][d].resize(Q);
    DF[lev][d].resize(Q);
    DFm[lev][d].resize(Q);
    DFp[lev][d].resize(Q);

    H_p[lev][d].resize(Q);
    H_m[lev][d].resize(Q);
    Fnum[lev][d].resize(Q);
    Fnumm_int[lev][d].resize(Q);
    Fnump_int[lev][d].resize(Q);
  }
  
  for(int q=0; q<Q; ++q){
    InitData_component(lev, ba,dm,q); 
  } 
}

void AmrDG::InitData_component(int lev,const BoxArray& ba,
                              const DistributionMapping& dm, int q)
{
  //Init data for given level for specific solution component
  AllPrint() <<"AmrDG::InitData_component() "<< q<<"\n";
   
  U_w[lev][q].define(ba, dm, Np, nghost);
  U_w[lev][q].setVal(0.0);

  amrex::BoxArray c_ba = U_w[lev][q].boxArray();

  H_w[lev][q].define(ba, dm, mNp, nghost);
  H_w[lev][q].setVal(0.0);
  
  U[lev][q].define(ba, dm, qMp, nghost);
  U[lev][q].setVal(0.0);
    
  H[lev][q].define(ba, dm, qMp, nghost);
  H[lev][q].setVal(0.0);
  
  U_center[lev][q].define(ba, dm, 1, nghost);
  U_center[lev][q].setVal(0.0);
       
  if(sim->model_pde->flag_source_term){S[lev][q].define(ba, dm, qMp, nghost);
  S[lev][q].setVal(0.0);}
    
  idc_curl_K[lev].define(ba, dm,1,0);
  idc_curl_K[lev].setVal(0.0);
  idc_div_K[lev].define(ba, dm,1,0);
  idc_div_K[lev].setVal(0.0);
  idc_grad_K[lev].define(ba, dm,1,0);
  idc_grad_K[lev].setVal(0.0);
    
  for(int d=0; d<AMREX_SPACEDIM; ++d){ 
    F[lev][d][q].define(ba, dm,qMp,nghost);
    F[lev][d][q].setVal(0.0);
    DF[lev][d][q].define(ba, dm,qMp,nghost);
    DF[lev][d][q].setVal(0.0);
    
    H_p[lev][d][q].define(ba, dm,qMpbd,nghost);
    H_p[lev][d][q].setVal(0.0);
    H_m[lev][d][q].define(ba, dm,qMpbd,nghost);
    H_m[lev][d][q].setVal(0.0);
    Fm[lev][d][q].define(ba, dm,qMpbd,nghost);
    Fm[lev][d][q].setVal(0.0);
    Fp[lev][d][q].define(ba, dm,qMpbd,nghost);
    Fp[lev][d][q].setVal(0.0);
    DFm[lev][d][q].define(ba, dm,qMpbd,nghost);
    DFm[lev][d][q].setVal(0.0);
    DFp[lev][d][q].define(ba, dm,qMpbd,nghost);
    DFp[lev][d][q].setVal(0.0);
     
    Fnumm_int[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,Np,0);    
    Fnumm_int[lev][d][q].setVal(0.0);
    
    Fnump_int[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,Np,0);    
    Fnump_int[lev][d][q].setVal(0.0);  
    
    Fnum[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,qMpbd,0);    
    Fnum[lev][d][q].setVal(0.0);     
  }
}

void AmrDG::ClearLevel(int lev) 
{
  Print() << "ClearLevel   "<< lev<<"\n";
  U_w[lev].clear();  
  U[lev].clear();  
  H_w[lev].clear();  
  H[lev].clear();  
  if(sim->model_pde->flag_source_term){S[lev].clear();}
  U_center[lev].clear();  
  
  idc_curl_K[lev].clear();
  idc_div_K[lev].clear();
  idc_grad_K[lev].clear();
  
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    H_p[lev][d].clear(); 
    H_m[lev][d].clear();
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

void AmrDG::RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm)
{
  Print() << "RemakeLevel   "<< lev<<"\n";
  
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
}

void AmrDG::FillPatch(int lev, Real time, amrex::MultiFab& mf,int icomp, int ncomp, int q)
{  
  Print() << "FillPatch   "<< lev<< " |component  "<<q<<"\n";
  //exchange internal and external ghost cells data for a single level for U_w data, i
  //if external ghost cells overlap a coarser one
  //an interpolation is made
  //NB: boundary conditions are not applied here
  if (lev == 0)
  { 
    amrex::Vector<MultiFab*> smf;
    amrex::Vector<Real> stime;
    GetData(lev, q,time, smf, stime);
    
    amrex::CpuBndryFuncFab bcf(nullptr); 
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> physbcf(geom[lev],bc_w[q],bcf);
    
    amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,geom[lev], physbcf, 0);  
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

  }
}

void AmrDG::MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm)
{
  Print() << "make new level from coarse :   "<< lev<< "\n";
  InitData_system(lev,ba,dm); 
  for(int q=0 ; q<Q; ++q){
    FillCoarsePatch(lev, time, U_w[lev][q], 0, Np,q);
  }
}

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

void AmrDG::AverageFineToCoarse()
{  
  Print() << "AverageFineToCoarse()"<< "\n";
  //averages data from finer cells to the respective covered coarse cell
  for (int l = finest_level; l > 0; --l){  
    for(int q=0; q<Q; ++q){   
      custom_interp.average_down(U_w[l][q], U_w[l-1][q],0,U_w[l-1][q].nComp(), refRatio(l-1), l,l-1);
    }
  } 
}

void AmrDG::Evolve()
{
  Print()<<"initial error norm "<<"\n";
  NormDG();
  int n=0;
  amrex::Real t= 0.0;  
  if(t_outplt>0){PlotFile(0, t);}
  
  ComputeDt();

  Print().SetPrecision(6)<<"time: "<< t<<" | time step: "<<n<<" | step size: "<< dt<<"\n";
  Print()<<"------------------------------------------------"<<"\n";
    
  while(t<T)
  {  
    if ((max_level > 0) && (n>0))
    {
      if((t_regrid > 0) && (n % t_regrid == 0)){
        regrid(0, t);
      }
    }
    
    //advance solution by one time-step.
    Time_Integration();
    //limit solution
    if((t_limit>0) && (n%t_limit==0)){Limiter_w(finest_level);}

    //gather valid fine cell solutions U_w into valid coarse cells
    AverageFineToCoarse();
     
    //sync ghost cells of U_w across same level and for coarse-fine interfaces
    //across different levels. NB: if regrid will happen at next timestep, 
    //then this procedure will be repeated again
    for(int l=0; l<=finest_level; ++l){
      for(int q=0; q<Q; ++q){ 
        FillPatch(l, t, U_w[l][q], 0, Np,q);  
      }
    } 

    n+=1;
    t+=dt;
    Print().SetPrecision(6)<<"time: "<< t<<" | time step: "<<n<<" | step size: "<< dt<<"\n";
    Print()<<"------------------------------------------------"<<"\n";
    
    if((t_outplt>0) && (n%t_outplt==0)){PlotFile(n, t);} 
    
    ComputeDt();
    if(T-t<dt){dt = T-t;}   
  } 
  
  NormDG();
}

void AmrDG::Time_Integration()
{ 
  Print() << "Time_Integraton()"<< "\n";
  ADER();
}

void AmrDG::ADER()
{ 
  for(int l=0; l<=finest_level; ++l){
  
    //Boundary Conditions on Modes U_w
    FillBoundaryCells(&(U_w[l]), l);
    
    //DG-PREDICTOR H_w, H
    //init first Np modes of H_w using U_w  
    Predictor_set(&(U_w[l]), &(H_w[l]));  
    
    //iteratively find best predictor (local to each cell)
    int iter=0;    
    while(iter<p)
    {
      //  s(H)
      if(sim->model_pde->flag_source_term){Source(l, qMp, &(H_w[l]), &(H[l]),&(S[l]),xi_ref_GLquad,true);}  
      //  f(H)
      for(int d = 0; d<AMREX_SPACEDIM; ++d){
        Flux(l,d,qMp,&(H_w[l]),&(H[l]),&(F[l][d]),&(DF[l][d]),xi_ref_GLquad, false, true);
      }   
      
      //compute new H_w
      for(int q=0; q<Q; ++q){
        Update_H_w(l, q);
      }
      iter+=1;
    }
    
    //DG-CORRECTOR U_w, U
    //Boundary fluxes F stored cell centered
    //Numerical boundary fluxes stored at cell interfaces
    //Intsegral value of Numeical fluxes stored at cell centers  
    //  s(H)
    if(sim->model_pde->flag_source_term){Source(l, qMp, &(H_w[l]), &(H[l]),&(S[l]),xi_ref_GLquad,true);}     
    for(int d = 0; d<AMREX_SPACEDIM; ++d){
      //  f(H)
      Flux(l,d,qMp,&(H_w[l]),&(H[l]),&(F[l][d]),&(DF[l][d]),xi_ref_GLquad, false, true);
      //  f(H_m)
      Flux(l,d,qMpbd,&(H_w[l]),&(H_m[l][d]),&(Fm[l][d]),&(DFm[l][d]),xi_ref_GLquad_bdm[d], true, true);
      //  f(H_P)
      Flux(l,d,qMpbd,&(H_w[l]),&(H_p[l][d]),&(Fp[l][d]),&(DFp[l][d]),xi_ref_GLquad_bdp[d], true, true);    
      //  fnum(f(H_p,i),f(H_m,i+1),H_p(i),H_m(i+1))      
      InterfaceNumFlux(l,d,qMpbd,&(H_m[l][d]),&(H_p[l][d]));
    } 
  }

  for (int l = finest_level; l > 0; --l){
    for(int d = 0; d<AMREX_SPACEDIM; ++d){
      for(int q=0; q<Q; ++q){        
        custom_interp.average_down(Fnumm_int[l][d][q], Fnumm_int[l-1][d][q],0,
                                  Fnumm_int[l-1][d][q].nComp(), refRatio(l-1), 
                                  l,l-1,d,true);
        custom_interp.average_down(Fnump_int[l][d][q], Fnump_int[l-1][d][q],0,
                                  Fnump_int[l-1][d][q].nComp(), refRatio(l-1), 
                                  l,l-1,d,true);  
                               
        //custom_interp.average_down(Fnum[l][d][q], Fnum[l-1][d][q],0,
        //                          Fnum[l-1][d][q].nComp(), refRatio(l-1), 
        //                          l,l-1,d,true); 
      }
    }
  }
  
  //Update solution and sync between grid levels
  //only validbox will have the correct values of the new U_w 
  for(int l=0; l<=finest_level; ++l){
    for(int q=0; q<Q; ++q){
      Update_U_w(l,q);    
    }
  }  
}

void AmrDG::Update_H_w(int lev, int q)
{ 
  if(sim->model_pde->flag_source_term){
  
    amrex::MultiFab& state_h_w = H_w[lev][q];
    amrex::MultiFab& state_u_w = U_w[lev][q];
    amrex::MultiFab& state_source = S[lev][q];
    
    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM); 
    
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F[lev][d][q]); 
    }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
      amrex::Vector<const amrex::FArrayBox *> fab_f(AMREX_SPACEDIM);
      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);  
    
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(state_h_w,MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(state_h_w,true); mfi.isValid(); ++mfi)
      #endif  
      {
        const amrex::Box& bx = mfi.growntilebox();
        
        amrex::FArrayBox& fab_h_w = state_h_w[mfi];
        amrex::Array4<amrex::Real> const& hw = fab_h_w.array();

        amrex::FArrayBox& fab_u_w = state_u_w[mfi];
        amrex::Array4<amrex::Real> const& uw = fab_u_w.array();
        
        amrex::FArrayBox& fab_source = state_source[mfi];
        amrex::Array4<amrex::Real> const& source = fab_source.array();
        
        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          fab_f[d]=state_f[d]->fabPtr(mfi);
          f[d]= fab_f[d]->const_array();
        } 
        
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          auto const dx = geom[lev].CellSizeArray();
          amrex::Vector<amrex::Real> rhs(mNp);
          for(int n = 0; n<mNp; ++n){  
            rhs[n] = 0.0;
            
            for(int m =0; m<Np; ++m)
            {         
              rhs[n] += Mk_pred[n][m]*uw(i,j,k,m);           
            } 
                       
            for(int d=0; d<AMREX_SPACEDIM; ++d)
            {
              for(int m =0; m<qMp; ++m)
              {
                rhs[n] -= ((dt/(amrex::Real)dx[d])*Sk_predVinv[d][n][m]*f[d](i,j,k,m));  
              }
            } 
            
            for(int m =0; m<qMp; ++m){
              rhs[n]+=(dt/2.0)*Mk_sVinv[n][m]*source(i,j,k,m);
            }  
          }

          for(int n = 0; n<mNp; ++n){ 
            amrex::Real sum = 0.0;
            for(int m =0; m<mNp; ++m){
              sum += Mk_h_w_inv[n][m]*rhs[m];
            }
            hw(i,j,k,n) = sum;
          }  
        });
      }
    }    
  }
  else
  {
    amrex::MultiFab& state_h_w = H_w[lev][q];
    amrex::MultiFab& state_u_w = U_w[lev][q];
    
    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM);  

    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F[lev][d][q]); 
    }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
    {
      amrex::Vector<const amrex::FArrayBox *> fab_f(AMREX_SPACEDIM);
      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);  
    
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(state_h_w,MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(state_h_w,true); mfi.isValid(); ++mfi)
      #endif  
      {
        const amrex::Box& bx = mfi.growntilebox();
        
        amrex::FArrayBox& fab_h_w = state_h_w[mfi];
        amrex::Array4<amrex::Real> const& hw = fab_h_w.array();

        amrex::FArrayBox& fab_u_w = state_u_w[mfi];
        amrex::Array4<amrex::Real> const& uw = fab_u_w.array();
        
        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          fab_f[d]=state_f[d]->fabPtr(mfi);
          f[d]= fab_f[d]->const_array();
        } 
        
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          auto const dx = geom[lev].CellSizeArray();
          amrex::Vector<amrex::Real> rhs(mNp);
          for(int n = 0; n<mNp; ++n){  
            rhs[n] = 0.0;
            
            for(int m =0; m<Np; ++m)
            {         
              rhs[n] += Mk_pred[n][m]*uw(i,j,k,m);           
            } 
       
            for(int d=0; d<AMREX_SPACEDIM; ++d)
            {
              for(int m =0; m<qMp; ++m)
              {
                rhs[n] -= ((dt/(amrex::Real)dx[d])*Sk_predVinv[d][n][m]*f[d](i,j,k,m));  
              }
            }            
          }

          for(int n = 0; n<mNp; ++n){ 
            amrex::Real sum = 0.0;
            for(int m =0; m<mNp; ++m){
              sum += Mk_h_w_inv[n][m]*rhs[m];
            }
            hw(i,j,k,n) = sum;
          }           
        });
      }
    }     
  }
}

            
void AmrDG::Update_U_w(int lev, int q)
{ 
  if(sim->model_pde->flag_source_term)
  {  
    amrex::MultiFab& state_u_w = U_w[lev][q];
    amrex::MultiFab& state_source = S[lev][q];
    
    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM); 
    amrex::Vector<const amrex::MultiFab *> state_fnum(AMREX_SPACEDIM); 
    amrex::Vector<const amrex::MultiFab *> state_fnumm_int(AMREX_SPACEDIM); 
    amrex::Vector<const amrex::MultiFab *> state_fnump_int(AMREX_SPACEDIM);
    
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F[lev][d][q]); 
      state_fnum[d]=&(Fnum[lev][d][q]); 
      state_fnumm_int[d]=&(Fnumm_int[lev][d][q]); 
      state_fnump_int[d]=&(Fnump_int[lev][d][q]);
    }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
    { 
      amrex::Vector<const amrex::FArrayBox *> fab_f(AMREX_SPACEDIM);
      amrex::Vector<const amrex::FArrayBox *> fab_fnum(AMREX_SPACEDIM);  
      amrex::Vector<const amrex::FArrayBox *> fab_fnumm_int(AMREX_SPACEDIM);
      amrex::Vector<const amrex::FArrayBox *> fab_fnump_int(AMREX_SPACEDIM);
      
      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);   
      amrex::Vector<amrex::Array4<const amrex::Real> > fnum(AMREX_SPACEDIM); 
      amrex::Vector<amrex::Array4<const amrex::Real> > fnumm_int(AMREX_SPACEDIM); 
      amrex::Vector<amrex::Array4<const amrex::Real> > fnump_int(AMREX_SPACEDIM); 
      
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(state_u_w,MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(state_u_w,true); mfi.isValid(); ++mfi)
      #endif  
      {
        const amrex::Box& bx = mfi.tilebox();
        
        amrex::FArrayBox& fab_u_w = state_u_w[mfi];
        amrex::Array4<amrex::Real> const& uw = fab_u_w.array();
        
        amrex::FArrayBox& fab_source = state_source[mfi];
        amrex::Array4<amrex::Real> const& source = fab_source.array();
        
        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          fab_f[d]=state_f[d]->fabPtr(mfi);
          fab_fnum[d]=state_fnum[d]->fabPtr(mfi);
          fab_fnumm_int[d]=state_fnumm_int[d]->fabPtr(mfi);
          fab_fnump_int[d]=state_fnump_int[d]->fabPtr(mfi);
          
          f[d]= fab_f[d]->const_array();
          fnum[d]= fab_fnum[d]->const_array();
          fnumm_int[d]= fab_fnumm_int[d]->const_array();
          fnump_int[d]= fab_fnump_int[d]->const_array();
        } 
        
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          auto const dx = geom[lev].CellSizeArray();     

          amrex::Real S_norm; 
          amrex::Real Mbd_norm; 
          int shift[] = {0,0,0};
          for(int n = 0; n<Np; ++n){
            amrex::Real rhs = 0.0;
            rhs+=(Mk_corr[n][n]*uw(i,j,k,n));
            
            for  (int d = 0; d < AMREX_SPACEDIM; ++d){
              S_norm= (dt/(amrex::Real)dx[d]);        
              for  (int m = 0; m < qMp; ++m){ 
                rhs+=S_norm*(Sk_corr[d][n][m]*((f)[d])(i,j,k,m));
              }
            }
  
            for  (int d = 0; d < AMREX_SPACEDIM; ++d){
              shift[d] = 1;
              Mbd_norm =  (dt/(amrex::Real)dx[d]);
              rhs-=(Mbd_norm*((fnump_int)[d])(i+shift[0],j+shift[1],k+shift[2],n));
              rhs-=(-Mbd_norm*((fnumm_int)[d])(i,j,k,n));
              shift[d] = 0;
            }
            
            for  (int m = 0; m < qMp; ++m){
              rhs+=((dt/2.0)*volquadmat[n][m]*source(i,j,k,m));
            }
            
            rhs/=Mk_corr[n][n];
            uw(i,j,k,n) = rhs;
          }       
        });
      }
    }  
  }
  else
  {
    amrex::MultiFab& state_u_w = U_w[lev][q];

    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM); 
    amrex::Vector<const amrex::MultiFab *> state_fnum(AMREX_SPACEDIM); 
    amrex::Vector<const amrex::MultiFab *> state_fnumm_int(AMREX_SPACEDIM); 
    amrex::Vector<const amrex::MultiFab *> state_fnump_int(AMREX_SPACEDIM); 
    
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F[lev][d][q]); 
      state_fnum[d]=&(Fnum[lev][d][q]); 
      state_fnumm_int[d]=&(Fnumm_int[lev][d][q]); 
      state_fnump_int[d]=&(Fnump_int[lev][d][q]); 
    }
    
    auto const dx = geom[lev].CellSizeArray();  
    amrex::Real vol = 1.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      vol*=dx[d];
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
    {
      amrex::Vector<const amrex::FArrayBox *> fab_f(AMREX_SPACEDIM);
      amrex::Vector<const amrex::FArrayBox *> fab_fnum(AMREX_SPACEDIM);
      amrex::Vector<const amrex::FArrayBox *> fab_fnumm_int(AMREX_SPACEDIM);
      amrex::Vector<const amrex::FArrayBox *> fab_fnump_int(AMREX_SPACEDIM);
      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);   
      amrex::Vector<amrex::Array4<const amrex::Real> > fnum(AMREX_SPACEDIM); 
      amrex::Vector<amrex::Array4<const amrex::Real> > fnumm_int(AMREX_SPACEDIM); 
      amrex::Vector<amrex::Array4<const amrex::Real> > fnump_int(AMREX_SPACEDIM); 
      
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(state_u_w,MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(state_u_w,true); mfi.isValid(); ++mfi)
      #endif 
      {
        const amrex::Box& bx = mfi.tilebox();
        
        amrex::FArrayBox& fab_u_w = state_u_w[mfi];
        amrex::Array4<amrex::Real> const& uw = fab_u_w.array();
                 
        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          fab_f[d]=state_f[d]->fabPtr(mfi);
          fab_fnum[d]=state_fnum[d]->fabPtr(mfi);
          fab_fnumm_int[d]=state_fnumm_int[d]->fabPtr(mfi);
          fab_fnump_int[d]=state_fnump_int[d]->fabPtr(mfi);
          
          f[d]= fab_f[d]->const_array();
          fnum[d]= fab_fnum[d]->const_array();
          fnumm_int[d]= fab_fnumm_int[d]->const_array();
          fnump_int[d]= fab_fnump_int[d]->const_array();
        } 
        
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          amrex::Real S_norm; 
          amrex::Real Mbd_norm; 
          int shift[] = {0,0,0};
          for(int n = 0; n<Np; ++n){
            amrex::Real rhs = 0.0;
            rhs+=(Mk_corr[n][n]*uw(i,j,k,n));
            
            for  (int d = 0; d < AMREX_SPACEDIM; ++d){
              S_norm= (dt/(amrex::Real)dx[d]);        
              for  (int m = 0; m < qMp; ++m){ 
                rhs+=S_norm*(Sk_corr[d][n][m]*((f)[d])(i,j,k,m));
              }
            }

            ///*
            for  (int d = 0; d < AMREX_SPACEDIM; ++d){
              shift[d] = 1;
              Mbd_norm =  (dt/(amrex::Real)dx[d]);
              //Mbd_norm =  dt/vol;
              rhs-=(Mbd_norm*((fnump_int)[d])(i+shift[0],j+shift[1],k+shift[2],n));
              rhs-=(-Mbd_norm*((fnumm_int)[d])(i,j,k,n));
              shift[d] = 0;
            }
            //*/

            /*
            for  (int d = 0; d < AMREX_SPACEDIM; ++d){
              shift[d] = 1;
              Mbd_norm =  (dt/(amrex::Real)dx[d]);
              for  (int m = 0; m < qMpbd; ++m){ 
                rhs-=(Mbd_norm*(Mkbd[2*d+1][n][m]*((fnum)[d])(i+shift[0],j+shift[1],
                      k+shift[2],m)-Mkbd[2*d][n][m]*((fnum)[d])(i,j,k,m)));   
              }
              shift[d] = 0;
            }  
            */
            
            rhs/=Mk_corr[n][n];
            uw(i,j,k,n) = rhs;
          }
        });
      }
    }    
  }
}
 
void AmrDG::ComputeDt()
{
  //compute minimum time step size s.t CFL condition is met
  amrex::Real safety_factor = CFL;
  
  //construt a center point
  amrex::Vector<amrex::Real> xi_ref_center(AMREX_SPACEDIM);
  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    xi_ref_center[d]=0.0;
  }

  amrex::Vector<amrex::Real> dt_tmp(finest_level+1);

  for (int l = 0; l <= finest_level; ++l)
  {
    auto const dx = geom[l].CellSizeArray();
    //compute average mesh size
    amrex::Real dx_avg = 0.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      dx_avg+=((amrex::Real)dx[d]/(amrex::Real)AMREX_SPACEDIM);
    }
      
    //evaluate modes at cell center pt
    get_U_from_U_w(0,&(U_w[l]),&(U_center[l]), xi_ref_center,false);
           
    //vector to accumulate all the min dt computed by this rank
    amrex::Vector<amrex::Real> rank_min_dt;

    amrex::Vector<const amrex::MultiFab *> state_uc(Q);

    for(int q=0; q<Q; ++q){
      state_uc[q]= &(U_center[l][q]);
    } 
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
      amrex::Vector<const amrex::FArrayBox *> fab_uc(Q);
      amrex::Vector<amrex::Array4<const amrex::Real>> uc(Q);
    
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(U_center[l][0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(U_center[l][0],true); mfi.isValid(); ++mfi)
      #endif 
      {
        const amrex::Box& bx = mfi.tilebox();
        
        for(int q=0; q<Q; ++q){
          fab_uc[q] = state_uc[q]->fabPtr(mfi);
          uc[q] =fab_uc[q]->const_array();
        }              
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          //compute max signal velocity lambda_max(for scalar case is derivative 
          //of flux, for system case is eigenvalue of flux jacobian
          amrex::Real lambda_max = 0.0;

          amrex::Vector<amrex::Real> lambda_d(AMREX_SPACEDIM);
          for(int d = 0; d < AMREX_SPACEDIM; ++d){  
            //compute at cell center so m==0
            lambda_d.push_back(sim->model_pde->pde_CFL(d,0,i,j,k,&uc));
          }
          //find max signal speed across the dimensions
          auto lambda_max_  = std::max_element(lambda_d.begin(),lambda_d.end());
          lambda_max = static_cast<amrex::Real>(*lambda_max_);         

          //general CFL formulation
          amrex::Real dt_cfl = (1.0/(2.0*(amrex::Real)p+1.0))
                        *(1.0/(amrex::Real)AMREX_SPACEDIM)*(dx_avg/lambda_max);
                        
          #pragma omp critical
          {
            rank_min_dt.push_back(safety_factor*dt_cfl);
          }
        });         
      }
    }   
    amrex::Real rank_lev_dt_min= 1.0;
    if (!rank_min_dt.empty()) {
      //compute the min in this rank for this level   
      auto rank_lev_dt_min_ = std::min_element(rank_min_dt.begin(), rank_min_dt.end());

      if (rank_lev_dt_min_ != rank_min_dt.end()) {
        rank_lev_dt_min = static_cast<amrex::Real>(*rank_lev_dt_min_);
      }
    }
    ParallelDescriptor::ReduceRealMin(rank_lev_dt_min);//, dt_tmp.size());
    dt_tmp[l] = rank_lev_dt_min;
    
  }
  amrex::Real dt_min = 1.0;
  if (!dt_tmp.empty()) {
    //min across levels
    auto dt_min_  = std::min_element(dt_tmp.begin(),dt_tmp.end());
    
    if (dt_min_ != dt_tmp.end()) {
      dt_min = (amrex::Real)(*dt_min_);//static_cast<amrex::Real>(*dt_min_); 
    }
  }
  ParallelDescriptor::ReduceRealMin(dt_min);
  dt = dt_min; 
}

