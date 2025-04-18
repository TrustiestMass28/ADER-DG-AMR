#include "AmrDG.h"
//#include "ModelEquation.h"

using namespace amrex;

void AmrDG::DEBUG_print_MFab() 
{ 
  //this function is used for debugging and prints out the specified MFab 
  //and if wanted also multilevel data. Is jsut a cleaner option
  //than copy paste the loop in the already dense code
  //user should implement wathever they want
  int q = 0;
  int lev = 0;
  int dim = 0;

  int nproc = amrex::ParallelDescriptor::MyProc();
  int nprocs = amrex::ParallelDescriptor::NProcs();
  AllPrint() << nproc<<"\n";
  amrex::MultiFab& state_c = U_w[lev][q];
  //amrex::MultiFab& state_c = H_w[lev][q];
  //amrex::MultiFab& state_c = Fnum[lev][dim][q];
  //amrex::MultiFab& state_c = Fnumm_int[lev][dim][q];
  //amrex::MultiFab& state_c = Fp[lev][dim][q];
  //amrex::MultiFab& state_c = H_m[lev][dim][q];
  //auto ba_tmp = Fnumm_int[lev][dim][q].boxArray();

  for (MFIter mfi(state_c); mfi.isValid(); ++mfi){

    //const amrex::Box& bx = mfi.tilebox();
    const amrex::Box& bx = mfi.growntilebox();
    
    amrex::FArrayBox& fabc= state_c[mfi];
    amrex::Array4<amrex::Real> const& uc = fabc.array();
      
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);   
    
    for(int k = lo.z; k <= hi.z; ++k){  
      for(int i = lo.x; i <= hi.x; ++i){ 
        for(int j = lo.y; j <= hi.y; ++j){
          //for(int n = 0; n<qMpbd; ++n) {
          //for(int n = 0; n<basefunc->Np_s; ++n) {
          for(int n = 0; n<2; ++n) {
            amrex::Print(nproc) << "Rank " << nproc << ": " 
                                << "i=" << i << ", j=" << j << ", k=" << k
                                << ", w=" << n << ", val=" << uc(i,j,k,n) << "\n";
          }       
        } 
      }       
    }
  }
  // Barrier to make sure output is clean
  amrex::ParallelDescriptor::Barrier();
}

void AmrDG::settings(int _p, amrex::Real _T) {
  p = _p;
  T = _T;
}

void AmrDG::init()
{
  amrex::Vector<std::string> logo = {
    "######################################",
    "     _   __  __ ___     ___   ___ ",
    "    /_\\ |  \\/  | _ \\___|   \\ / __|",
    "   / _ \\| |\\/| |   /___| |) | (_ |",
    "  /_/ \\_\\_|  |_|_|_\\   |___/ \\___|",
    "",
    "       +..+..+----+---------+",
    "     ->|  |  |    |<==      |",
    "       +..+..+----|         |",
    "     ->|  |  |    |<==      |",
    "       +----+..+..+----+----+",
    "    ==>|    |  |  |    |    |",
    "       +----+..+..+----+----+",
    "     =>|    |  |  |    |    |",
    "       +----+..+..+----+----+",
    "                                      ",
    "      Adaptive Mesh Refinement      ",
    "                 &                  ",
    "       Discontinuous Galerkin       ",
    "######################################"
  };

  // Loop through the vector and print each line
  for (const auto& line : logo) {
    Print() << line << std::endl;
  }

  auto _mesh = mesh.lock();

  //Set vectors size
  U_w.resize(_mesh->L); 
  U.resize(_mesh->L); 
  if(flag_source_term){S.resize(_mesh->L);}
  U_center.resize(_mesh->L); 

  F.resize(_mesh->L);
  Fm.resize(_mesh->L);
  Fp.resize(_mesh->L);

  DF.resize(_mesh->L);
  DFm.resize(_mesh->L);
  DFp.resize(_mesh->L);

  Fnum.resize(_mesh->L);
  Fnumm_int.resize(_mesh->L);
  Fnump_int.resize(_mesh->L);

  H_w.resize(_mesh->L); 
  H.resize(_mesh->L); 
  H_p.resize(_mesh->L);
  H_m.resize(_mesh->L);

  //Basis function
  basefunc = std::make_shared<BasisLegendre>();

  //basefunc->setNumericalMethod(this);
  basefunc->setNumericalMethod(shared_from_this());
  
  //Number of modes/components of solution decomposition
  basefunc->set_number_basis();

  //basis functions d.o.f idx mapper
  basefunc->basis_idx_s.resize(basefunc->Np_s, amrex::Vector<int>(AMREX_SPACEDIM));
  basefunc->basis_idx_t.resize(basefunc->Np_st, amrex::Vector<int>(1));
  basefunc->basis_idx_st.resize(basefunc->Np_st, amrex::Vector<int>(AMREX_SPACEDIM+1));  
  //basis_idx_t for each of the Np_st basis func, stores the idx of the time polynomial,
  //thats why it has basefunc->Np_st. This because when we ened to evalaute temporal basis
  //it is always in same loop as other spatial component, i.e when looping over Np_st

  basefunc->set_idx_mapping_s();

  basefunc->set_idx_mapping_st();

  
  //Set-up quadrature rule
  quadrule = std::make_shared<QuadratureGaussLegendre>();

  quadrule->setNumericalMethod(shared_from_this());

  //Number of quadrature pts
  quadrule->set_number_quadpoints();

  //Init data structure holdin quadrature data
  quadrule->xi_ref_quad_s.resize(quadrule->qMp_s,amrex::Vector<amrex::Real> (AMREX_SPACEDIM));
  quadrule->xi_ref_quad_t.resize(quadrule->qMp_t,amrex::Vector<amrex::Real> (1)); 
  quadrule->xi_ref_quad_st.resize(quadrule->qMp_st,amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1));  
  quadrule->xi_ref_quad_st_bdm.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (quadrule->qMp_st_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)));                    
  quadrule->xi_ref_quad_st_bdp.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (quadrule->qMp_st_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)));

  //construt a center point of a [1,1]^D cell
  quadrule->xi_ref_quad_s_center.resize(1,amrex::Vector<amrex::Real> (AMREX_SPACEDIM));

  //xi_ref_equidistant.resize(qMp,amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)); 

  //Generation of quadrature pts
  quadrule->set_quadpoints();
                              
  //Initialize generalized Vandermonde matrix (only volume, no boudnary version)
  //and their inverse
  V.resize(quadrule->qMp_st,amrex::Vector<amrex::Real> (basefunc->Np_st)); 
  Vinv.resize(basefunc->Np_st,amrex::Vector<amrex::Real> (quadrule->qMp_st));

  //Initialize L2 projeciton quadrature matrix
  quadmat.resize(basefunc->Np_s,amrex::Vector<amrex::Real>(quadrule->qMp_s));  

  //Initialize generalized Element matrices for ADER-DG corrector
  Mk_corr.resize(basefunc->Np_s,amrex::Vector<amrex::Real>(basefunc->Np_s));
  Sk_corr.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Vector<amrex::Real>>(basefunc->Np_s,
                amrex::Vector<amrex::Real>(quadrule->qMp_st)));
  Mkbdm.resize(AMREX_SPACEDIM, amrex::Vector<amrex::Vector<amrex::Real>>(basefunc->Np_s,
                amrex::Vector<amrex::Real>(quadrule->qMp_st_bd)));

  Mkbdp.resize(AMREX_SPACEDIM, amrex::Vector<amrex::Vector<amrex::Real>>(basefunc->Np_s,
    amrex::Vector<amrex::Real>(quadrule->qMp_st_bd)));

  Mk_corr_src.resize(basefunc->Np_s,amrex::Vector<amrex::Real>(quadrule->qMp_st));  

  //Initialize generalized Element matrices for ADER predictor
  Mk_h_w.resize(basefunc->Np_st,amrex::Vector<amrex::Real>(basefunc->Np_st));
  Mk_h_w_inv.resize(basefunc->Np_st,amrex::Vector<amrex::Real>(basefunc->Np_st));
  Mk_pred.resize(basefunc->Np_st,amrex::Vector<amrex::Real>(basefunc->Np_s));  
  Sk_pred.resize(AMREX_SPACEDIM, amrex::Vector<amrex::Vector<amrex::Real>>(basefunc->Np_st,
                                  amrex::Vector<amrex::Real>(basefunc->Np_st)));
  Mk_pred_src.resize(basefunc->Np_st,amrex::Vector<amrex::Real>(basefunc->Np_st));

  Sk_predVinv.resize(AMREX_SPACEDIM, amrex::Vector<amrex::Vector<amrex::Real>>(basefunc->Np_st,
                      amrex::Vector<amrex::Real>(quadrule->qMp_st)));
  Mk_pred_srcVinv.resize(basefunc->Np_st,amrex::Vector<amrex::Real>(quadrule->qMp_st));
    
  //Construct system matrices
  set_vandermat();
    
  set_ref_element_matrix();
  
  //TODO:Set-up mesh interpolation

  //TODO::Set-up limiting

}

void AmrDG::init_bc(amrex::Vector<amrex::Vector<amrex::BCRec>>& bc, int& n_comp)
{
  //since use modal DG we will apply BCs to individual spatial modes
  n_comp = basefunc->Np_s;
  bc.resize(Q,amrex::Vector<amrex::BCRec>(n_comp));

  //bc evaluated at all spatial quadrature point in ghost cells
  //then they will be projected back to n_comp modes
  n_pt_bc = quadrule->qMp_s;
}

amrex::Real AmrDG::setBC(const amrex::Vector<amrex::Real>& bc, int comp,int dcomp,int q, int lev)
{
  //BC projection u|bc->u_w|bc
  amrex::Real sum = 0.0;
  for(int m=0; m<quadrule->qMp_s; ++m)
  {
    sum+= quadmat[dcomp +comp][m]*bc[m];
  }

  sum /=refMat_phiphi(dcomp + comp,basefunc->basis_idx_s,dcomp + comp,basefunc->basis_idx_s);

  return sum;
}

AmrDG::~AmrDG(){
  //delete basefunc;
  //delete quadrule;
}

void AmrDG::set_init_data_system(int lev,const BoxArray& ba,
                                  const DistributionMapping& dm)
{
  //Init data structures for level for all solution components of the system
  U_w[lev].resize(Q); 
  U[lev].resize(Q);
  
  if(flag_source_term){S[lev].resize(Q);}
  
  U_center[lev].resize(Q); 
  
  H_w[lev].resize(Q); 
  H[lev].resize(Q);  
  H_p[lev].resize(AMREX_SPACEDIM);
  H_m[lev].resize(AMREX_SPACEDIM);

  F[lev].resize(AMREX_SPACEDIM);
  Fm[lev].resize(AMREX_SPACEDIM);
  Fp[lev].resize(AMREX_SPACEDIM);
  DF[lev].resize(AMREX_SPACEDIM);
  DFm[lev].resize(AMREX_SPACEDIM);
  DFp[lev].resize(AMREX_SPACEDIM);

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
    Fnum[lev][d].resize(Q);
    Fnumm_int[lev][d].resize(Q);
    Fnump_int[lev][d].resize(Q);

    H_p[lev][d].resize(Q);
    H_m[lev][d].resize(Q);
  }
}

//Init data for given level for specific solution component
void AmrDG::set_init_data_component(int lev,const BoxArray& ba,
                                    const DistributionMapping& dm, int q)
{
  auto _mesh = mesh.lock();

  H_w[lev][q].define(ba, dm, basefunc->Np_st, _mesh->nghost);
  H_w[lev][q].setVal(0.0);
  H[lev][q].define(ba, dm, quadrule->qMp_st, _mesh->nghost);
  H[lev][q].setVal(0.0);
  
  U_w[lev][q].define(ba, dm, basefunc->Np_s, _mesh->nghost);
  U_w[lev][q].setVal(0.0);

  U[lev][q].define(ba, dm, quadrule->qMp_st, _mesh->nghost);
  U[lev][q].setVal(0.0);


  U_center[lev][q].define(ba, dm, 1, _mesh->nghost);
  U_center[lev][q].setVal(0.0);

  if(flag_source_term){S[lev][q].define(ba, dm, quadrule->qMp_st, _mesh->nghost);
  S[lev][q].setVal(0.0);}

  //idc_curl_K[lev].define(ba, dm,1,0);
  //idc_curl_K[lev].setVal(0.0);
  //idc_div_K[lev].define(ba, dm,1,0);
  //idc_div_K[lev].setVal(0.0);
  //idc_grad_K[lev].define(ba, dm,1,0);
  //idc_grad_K[lev].setVal(0.0);
    
  for(int d=0; d<AMREX_SPACEDIM; ++d){ 
    H_p[lev][d][q].define(ba, dm,quadrule->qMp_st_bd,_mesh->nghost);
    H_p[lev][d][q].setVal(0.0);

    H_m[lev][d][q].define(ba, dm,quadrule->qMp_st_bd,_mesh->nghost);
    H_m[lev][d][q].setVal(0.0);


    F[lev][d][q].define(ba, dm,quadrule->qMp_st,_mesh->nghost);
    F[lev][d][q].setVal(0.0);

    DF[lev][d][q].define(ba, dm,quadrule->qMp_st,_mesh->nghost);
    DF[lev][d][q].setVal(0.0);

    Fm[lev][d][q].define(ba, dm,quadrule->qMp_st_bd,_mesh->nghost);
    Fm[lev][d][q].setVal(0.0);

    Fp[lev][d][q].define(ba, dm,quadrule->qMp_st_bd,_mesh->nghost);
    Fp[lev][d][q].setVal(0.0);

    DFm[lev][d][q].define(ba, dm,quadrule->qMp_st_bd,_mesh->nghost);
    DFm[lev][d][q].setVal(0.0);

    DFp[lev][d][q].define(ba, dm,quadrule->qMp_st_bd,_mesh->nghost);
    DFp[lev][d][q].setVal(0.0);

    Fnum[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,quadrule->qMp_st_bd,0);    
    Fnum[lev][d][q].setVal(0.0);  

    Fnumm_int[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,basefunc->Np_s,0);    
    Fnumm_int[lev][d][q].setVal(0.0);

    Fnump_int[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,basefunc->Np_s,0);    
    Fnump_int[lev][d][q].setVal(0.0);     
  }
}

//N=
//M=qM = xi.size();
void AmrDG::get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>* U_ptr,
                          amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                          const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //Can evalaute U_w (Np_s) at boundary using bd quadrature pts qM_s_bd or 
  //for the entire cell using qM_s
  //M=quadrule->qM_s;
  //N=basefunc->qM_s

  //int qM = xi.size();

  
    amrex::Vector<const amrex::MultiFab *> state_u_w(Q);   
    amrex::Vector<amrex::MultiFab *> state_u(Q); 

    for(int q=0; q<Q; ++q){
      state_u_w[q]=&((*U_w_ptr)[q]);    
      state_u[q]=&((*U_ptr)[q]);  
    } 

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
      amrex::Vector<const amrex::FArrayBox *> fab_u_w(Q);
      amrex::Vector< amrex::Array4<const amrex::Real>> uw(Q);  
      amrex::Vector<amrex::FArrayBox *> fab_u(Q);
      amrex::Vector<amrex::Array4<amrex::Real>> u(Q);  
    
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(*(state_u_w[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(*(state_u_w[0]),true); mfi.isValid(); ++mfi)
      #endif
      {
        const amrex::Box& bx = mfi.growntilebox();

        for(int q=0 ; q<Q; ++q){
          fab_u_w[q] = state_u_w[q]->fabPtr(mfi);
          uw[q] = fab_u_w[q]->const_array();
              
          fab_u[q]=&((*(state_u[q]))[mfi]);
          u[q]=(*(fab_u[q])).array();
        } 
        for(int q=0 ; q<Q; ++q){
          amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
          {
            for(int m = 0; m<M ; ++m){
              (u[q])(i,j,k,m)=0.0;
            }
          }); 

          amrex::ParallelFor(bx,N,[&] (int i, int j, int k,int n) noexcept
          { 
            for(int m = 0; m<M ; ++m){
              (u[q])(i,j,k,m)+=((uw[q])(i,j,k,n)*basefunc->phi_s(n,basefunc->basis_idx_s,xi[m])); 
            }
          });  
                
        }      
      }  
    }
  
}

void AmrDG::get_H_from_H_w(int M, int N,amrex::Vector<amrex::MultiFab>* H_ptr,
                          amrex::Vector<amrex::MultiFab>* H_w_ptr, 
                          const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //int qM = xi.size();

  //Can evalaute H_w (Np_st) at boundary using bd quadrature pts qM_st_bd or 
  //for the entire cell using qM_st

  
    amrex::Vector<const amrex::MultiFab *> state_u_w(Q);   
    amrex::Vector<amrex::MultiFab *> state_u(Q); 

    for(int q=0; q<Q; ++q){
      state_u_w[q]=&((*H_w_ptr)[q]);    
      state_u[q]=&((*H_ptr)[q]);  
    } 

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
      amrex::Vector<const amrex::FArrayBox *> fab_u_w(Q);
      amrex::Vector< amrex::Array4<const amrex::Real>> uw(Q);  
      amrex::Vector<amrex::FArrayBox *> fab_u(Q);
      amrex::Vector<amrex::Array4<amrex::Real>> u(Q);  

      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(*(state_u_w[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(*(state_u_w[0]),true); mfi.isValid(); ++mfi)
      #endif
      {
        const amrex::Box& bx = mfi.growntilebox();

        for(int q=0 ; q<Q; ++q){
          fab_u_w[q] = state_u_w[q]->fabPtr(mfi);
          uw[q] = fab_u_w[q]->const_array();

          fab_u[q]=&((*(state_u[q]))[mfi]);
          u[q]=(*(fab_u[q])).array();
        } 
        for(int q=0 ; q<Q; ++q){
          amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
          {
            for(int m = 0; m<M ; ++m){
              (u[q])(i,j,k,m)=0.0;
            }
          }); 
    
          amrex::ParallelFor(bx,N,[&] (int i, int j, int k,int n) noexcept
          { 
            for(int m = 0; m<M ; ++m){
              (u[q])(i,j,k,m)+=((uw[q])(i,j,k,n)*basefunc->phi_st(n,basefunc->basis_idx_st,xi[m])); 
            }
          });            
        }      
      }  
    }
  
}

void AmrDG::set_predictor(const amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                          amrex::Vector<amrex::MultiFab>* H_w_ptr)
{

  amrex::Vector<const amrex::MultiFab *> state_u_w(Q);   
  amrex::Vector<amrex::MultiFab *> state_h_w(Q); 

  for(int q=0; q<Q; ++q){
    state_u_w[q]=&((*U_w_ptr)[q]);
    state_h_w[q]=&((*H_w_ptr)[q]); 
  } 

  #ifdef AMREX_USE_OMP
  #pragma omp parallel
  #endif
  { 
    amrex::Vector<amrex::FArrayBox *> fab_h_w(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> hw(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_u_w(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> uw(Q);   

    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_h_w[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(*(state_h_w[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){      
        fab_u_w[q] = state_u_w[q]->fabPtr(mfi);
        fab_h_w[q]= &((*(state_h_w[q]))[mfi]);

        uw[q] = fab_u_w[q]->const_array();
        hw[q]=(*(fab_h_w[q])).array();
      }

      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx, basefunc->Np_st,[&] (int i, int j, int k, int n) noexcept
        {   
          if(n<basefunc->Np_s)
          {
            //set first Np_s modes of H_w equalt to U_w
            amrex::Real tmp = (uw[q])(i,j,k,n); 
            (hw[q])(i,j,k,n)=tmp;  
          } 
          else
          {
            (hw[q])(i,j,k,n)=0.0;
          }                
        }); 
      }      
    }  
  }  
}

void AmrDG::numflux(int lev,int d,int M, int N,
                    amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                    amrex::Vector<amrex::MultiFab>* U_ptr_p,
                    amrex::Vector<amrex::MultiFab>* F_ptr_m,
                    amrex::Vector<amrex::MultiFab>* F_ptr_p,
                    amrex::Vector<amrex::MultiFab>* DF_ptr_m,
                    amrex::Vector<amrex::MultiFab>* DF_ptr_p)
{
  //computes the numerical flux at the plus interface of a cell, i.e at idx i+1/2 
  amrex::Vector<amrex::MultiFab *> state_fnum(Q); 
  
  amrex::Vector<const amrex::MultiFab *> state_fm(Q);
  amrex::Vector<const amrex::MultiFab *> state_fp(Q); 
  amrex::Vector<const amrex::MultiFab *> state_dfm(Q);
  amrex::Vector<const amrex::MultiFab *> state_dfp(Q);
  amrex::Vector<const amrex::MultiFab *> state_um(Q);
  amrex::Vector<const amrex::MultiFab *> state_up(Q);

  for(int q=0 ; q<Q; ++q){
    state_um[q] = &((*U_ptr_m)[q]); 
    state_up[q] = &((*U_ptr_p)[q]); 

    state_fm[q] = &((*F_ptr_m)[q]);
    state_fp[q] = &((*F_ptr_p)[q]);

    state_dfm[q] = &((*DF_ptr_m)[q]);
    state_dfp[q] = &((*DF_ptr_p)[q]);

    state_fnum[q] = &(Fnum[lev][d][q]); 
  }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector<amrex::FArrayBox *> fab_fnum(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnum(Q);
    amrex::Vector<amrex::FArrayBox *> fab_fnumm_int(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnumm_int(Q);
    amrex::Vector<amrex::FArrayBox *> fab_fnump_int(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnump_int(Q);
    
    amrex::Vector<const amrex::FArrayBox *> fab_fm(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > fm(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_fp(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > fp(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_dfm(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > dfm(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_dfp(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > dfp(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_um(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > um(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_up(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > up(Q);
    
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_fnum[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_fnum[0]),true); mfi.isValid(); ++mfi)
    #endif    
    {
      //externally grown tilebox
      const amrex::Box& bx = mfi.tilebox();
      //grow the box only in the high-end along dimension d
      //can be done because each MPI process has up-to date values of outer and inner ghost cells
      //which are used only to compute the fluxes that will impact the tilebox (ot grown) of that 
      //process
      amrex::Box ibx = amrex::growHi(bx, d, 1);
      //
      
      for(int q=0 ; q<Q; ++q){
        fab_fnum[q]=&((*(state_fnum[q]))[mfi]);
  
        fab_fm[q] = state_fm[q]->fabPtr(mfi);
        fab_fp[q] = state_fp[q]->fabPtr(mfi);
        fab_dfm[q] = state_dfm[q]->fabPtr(mfi);
        fab_dfp[q] = state_dfp[q]->fabPtr(mfi);
        fab_um[q] = state_um[q]->fabPtr(mfi);
        fab_up[q] = state_up[q]->fabPtr(mfi);
        
        fnum[q]=(*(fab_fnum[q])).array();

        fm[q] = fab_fm[q]->const_array();
        fp[q] = fab_fp[q]->const_array();
        dfm[q] = fab_dfm[q]->const_array();
        dfp[q] = fab_dfp[q]->const_array();
        um[q] = fab_um[q]->const_array();
        up[q] = fab_up[q]->const_array();
      }
            
      for(int q=0 ; q<Q; ++q){
        //compute the pointwise evaluations of the numerical flux
        amrex::ParallelFor(ibx, M,[&] (int i, int j, int k, int m) noexcept
        {
          //check which indices it iterate across, i.e if last one is reachd
          fnum[q](i,j,k,m) = LLF_numflux(d,m,i,j,k,up[q],um[q],fp[q],fm[q],dfp[q],dfm[q]);  
        });   
      }
    }
  }
}

amrex::Real AmrDG::LLF_numflux(int d, int m,int i, int j, int k, 
  amrex::Array4<const amrex::Real> up, 
  amrex::Array4<const amrex::Real> um, 
  amrex::Array4<const amrex::Real> fp,
  amrex::Array4<const amrex::Real> fm,  
  amrex::Array4<const amrex::Real> dfp,
  amrex::Array4<const amrex::Real> dfm)
{
  
  //implementation of the numerical flux across interface
  //---------
  //    |        
  //  L | R      
  //    |
  //---------
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

//updates solution on valid cells
void AmrDG::update_U_w(int lev)
{
  auto _mesh = mesh.lock();

  const auto dx = _mesh->get_dx(lev);
  amrex::Real vol = _mesh->get_dvol(lev);

  for(int q=0; q<Q; ++q){
    amrex::MultiFab& state_u_w = U_w[lev][q];

    amrex::MultiFab state_rhs;
    state_rhs.define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), basefunc->Np_s, _mesh->nghost); 
    state_rhs.setVal(0.0);
    
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
  
        amrex::FArrayBox& fab_rhs = state_rhs[mfi];
        amrex::Array4<amrex::Real> const& rhs = fab_rhs.array();
        
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

        amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
        {
          rhs(i,j,k,n)+=(Mk_corr[n][n]*uw(i,j,k,n));      
        });

        amrex::Real S_norm; 
        amrex::Real Mbd_norm;  
        int shift[] = {0,0,0};
        for  (int d = 0; d < AMREX_SPACEDIM; ++d){
          S_norm= (Dt/(amrex::Real)dx[d]);    
          for  (int m = 0; m < quadrule->qMp_st; ++m){ 
            amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
            {
              rhs(i,j,k,n)+=S_norm*(Sk_corr[d][n][m]*((f)[d])(i,j,k,m));
            });
          }
          
          Mbd_norm =  (Dt/(amrex::Real)dx[d]);
          shift[d] = 1;
          for  (int m = 0; m < quadrule->qMp_st_bd; ++m){ 
            amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
            {
              rhs(i,j,k,n)-=(Mbd_norm*(Mkbdp[d][n][m]*((fnum)[d])(i+shift[0],j+shift[1], k+shift[2],m)
                                      -Mkbdm[d][n][m]*((fnum)[d])(i,j,k,m)));   
            });
          }
          shift[d] = 0;
        }

        if(flag_source_term)
        { 
          amrex::MultiFab& state_source = S[lev][q];
          amrex::FArrayBox& fab_source = state_source[mfi];
          amrex::Array4<amrex::Real> const& source = fab_source.array();
          for  (int m = 0; m < quadrule->qMp_st; ++m){
            amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
            {
              rhs(i,j,k,n)+=((Dt/2.0)*Mk_corr_src[n][m]*source(i,j,k,m));
            });
          }
        }

        amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
        {
          rhs(i,j,k,n)/=Mk_corr[n][n];
          uw(i,j,k,n) = rhs(i,j,k,n);       
        });
      }
    }    
  }
}

void AmrDG::update_H_w(int lev)
{ 
  auto _mesh = mesh.lock();

  for(int q=0; q<Q; ++q)
  {
    const auto dx = _mesh->get_dx(lev);
    
    amrex::MultiFab& state_h_w = H_w[lev][q];
    amrex::MultiFab& state_u_w = U_w[lev][q];
    
    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM);  

    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F[lev][d][q]); 
    }
    
    amrex::MultiFab state_rhs;
    state_rhs.define(H_w[lev][q].boxArray(), H_w[lev][q].DistributionMap(), basefunc->Np_st, _mesh->nghost); 
    state_rhs.setVal(0.0); 
    
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
        
        amrex::FArrayBox& fab_rhs = state_rhs[mfi];
        amrex::Array4<amrex::Real> const& rhs = fab_rhs.array();
        
        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          fab_f[d]=state_f[d]->fabPtr(mfi);
          f[d]= fab_f[d]->const_array();
        } 
        
        
        for(int m =0; m<basefunc->Np_s; ++m)
        { 
          amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
          {
            rhs(i,j,k,n) += Mk_pred[n][m]*uw(i,j,k,m);       
            
            hw(i,j,k,n) = 0.0;
          });
        }
        
        for(int d=0; d<AMREX_SPACEDIM; ++d)
        {
          for(int m =0; m<quadrule->qMp_st; ++m)
          {
            amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
            {  
              rhs(i,j,k,n) -= ((Dt/(amrex::Real)dx[d])*Sk_predVinv[d][n][m]*f[d](i,j,k,m));
            });
          }
        }
      
        if(flag_source_term){
          amrex::MultiFab& state_source = S[lev][q];
          amrex::FArrayBox& fab_source = state_source[mfi];
          amrex::Array4<amrex::Real> const& source = fab_source.array();
        
          for(int m =0; m<quadrule->qMp_st; ++m)
          {
            amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
            {        
              rhs(i,j,k,n)+=(Dt/2.0)*Mk_pred_srcVinv[n][m]*source(i,j,k,m);
            });
          }
        }
        
        for(int m =0; m<basefunc->Np_st; ++m){  
          amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
          {
            hw(i,j,k,n) += Mk_h_w_inv[n][m]*rhs(i,j,k,m); 
          });       
        }
      }
    }     
  }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
/*

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>

#include <AMReX_Interpolater.H>
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
#include <AMReX_FillPatcher.H>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif
BOUNDARY CONDITIONS

///////////////////////////////////////////////////////////////////////
LIMITING AND REFINING ADVANCED

  //std::string _limiter_type, 
  //amrex::Real _TVB_M,
  //amrex::Vector<amrex::Real> _AMR_TVB_C ,
  //amrex::Vector<amrex::Real> _AMR_curl_C, 
  //amrex::Vector<amrex::Real> _AMR_div_C,  
  //amrex::Vector<amrex::Real> _AMR_grad_C, 
  //amrex::Vector<amrex::Real> _AMR_sec_der_C,
  //amrex::Real _AMR_sec_der_indicator, 
  //amrex::Vector<amrex::Real> _AMR_C, 
  int _t_limit

  t_limit  = _t_limit;
  AMR_curl_C = _AMR_curl_C;
  AMR_div_C = _AMR_div_C;
  AMR_grad_C = _AMR_grad_C;
  AMR_sec_der_C = _AMR_sec_der_C;
  AMR_sec_der_indicator = _AMR_sec_der_indicator;
  AMR_TVB_C = _AMR_TVB_C;
  AMR_C = _AMR_C;
  
    limiter_type = _limiter_type;
  TVB_M = _TVB_M;

  idc_curl_K.resize(L);
  idc_div_K.resize(L);
  idc_grad_K.resize(L);

///////////////////////////////////////////////////////////////////////
AMR
  int _t_regrid, , 
 t_regrid = _t_regrid;

   //Refinement parameters fine tuning
  AMR_settings_tune();


///////////////////////////////////////////////////////////////////////
Mesh Interpolation


  //Interpolation coarse<->fine data scatter/gather
  custom_interp.getouterref(this); 
  custom_interp.interp_proj_mat();
  
*/


/*





//std::swap(U_w[lev][q],new_mf);    

/*
  //amrex::MultiFab new_mf;
//new_mf.define(ba, dm, Np, nghost);
//new_mf.setVal(0.0);    
  amrex::FillPatchTwoLevels(new_mf, time, cmf, ctime, fmf, ftime,0, 0, Np, 
                          geom[lev-1], geom[lev],coarse_physbcf, 0, fine_physbcf, 
                          0, refRatio(lev-1),mapper, bc_w[q], 0);
                          
                          
fillpatcher = std::make_unique<FillPatcher<MultiFab>>(ba, dm, geom[lev],
parent->boxArray(level-1), parent->DistributionMap(level-1), geom_crse,
IntVect(nghost), desc.nComp(), desc.interp(scomp));


fillpatcher->fill(mf, IntVect(nghost), time,
    smf_crse, stime_crse, smf_fine, stime_fine,
    scomp, dcomp, ncomp,
    physbcf_crse, scomp, physbcf_fine, scomp,
    desc.getBCs(), scomp);*/
