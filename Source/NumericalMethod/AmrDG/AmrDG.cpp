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
  int lev = 1;
  int dim = 0;
  int M = 1;

  int nproc = amrex::ParallelDescriptor::MyProc();
  int nprocs = amrex::ParallelDescriptor::NProcs();
  AllPrint() << nproc<<"\n";
  //amrex::MultiFab& state_c = F[lev][dim][q];
  //amrex::MultiFab& state_c = U_w[lev][q];
  //amrex::MultiFab& state_c = H_w[lev][q];
  //amrex::MultiFab& state_c = Fnum[lev][dim][q];
  amrex::MultiFab& state_c = DFp[lev][dim][q];
  //amrex::MultiFab& state_c = H_p[lev][dim][q];

  for (MFIter mfi(state_c); mfi.isValid(); ++mfi){

    //const amrex::Box& bx = mfi.tilebox();
    const amrex::Box& bx = mfi.growntilebox();
    
    amrex::FArrayBox& fabc= state_c[mfi];
    amrex::Array4<amrex::Real> const& uc = fabc.array();
      
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);   
    
      amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
      {    
        amrex::Print(nproc) << "Rank " << nproc << ": " 
                            << "i=" << i << ", j=" << j << ", k=" << k
                            << ", w=" << m << ", val=" << uc(i,j,k,m) << "\n";
      });

  }
}

void AmrDG::settings(int _p, amrex::Real _T) {
  p = _p;
  T = _T;
}


void AmrDG::AMR_advanced_settings()
{
  auto _mesh = mesh.lock();

  //Set the same blocking factor for all levels
  _mesh->SetBlockingFactor(2);
  _mesh->SetGridEff(0.9);

  //Different blocking factor for each refinemetn level
  //amrex::Vector<int> block_fct;// (max_level+1);
  //for (int l = 0; l <= max_level; ++l) {
  //  if(l==0){block_fct.push_back(8);}
  //  else if(l==1){block_fct.push_back(4);}
  //}
  ////NB: can also specify different block factor per dimension and different
  ////block factor per dimension per le

  //iterate_on_new_grids = false;//will genrete only one new level per refinement step
}

void AmrDG::init()
{ /*
  amrex::Vector<std::string> logo0 = {
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
  for (const auto& line : logo0) {
    Print() << line << std::endl;
  }*/

    const int width = 80;
    std::vector<std::string> logo = {
    "     _   __  __ ___     ___   ___ ",
    "    /_\\ |  \\/  | _ \\___|   \\ / __|",
    "   / _ \\| |\\/| |   /___| |) | (_ |",
    "  /_/ \\_\\_|  |_|_|_\\   |___/ \\___|",
    "",
    "    +..+..+----+---------+",
    "  ->|  |  |    |<==      |",
    "    +..+..+----|         |",
    "  ->|  |  |    |<==      |",
    "    +----+..+..+----+----+",
    " ==>|    |  |  |    |    |",
    "    +----+..+..+----+----+",
    "  =>|    |  |  |    |    |",
    "    +----+..+..+----+----+",
    "                                     ",
    "       Adaptive Mesh Refinement     ",
    "                  &                 ",
    "        Discontinuous Galerkin      ",
  };

    // Print top border (80 #)
    amrex::Print() << std::string(width, '#') << "\n";

    for (const auto& line : logo) {
        int offset = 3; // shift left by 5 spaces
        int padding = ((width - static_cast<int>(line.size())) / 2) - offset;
        // If padding < 0 means line longer than width, just print it as is
        if (padding < 0) {
            amrex::Print() << line << "\n";
        } else {
            // Left pad with spaces to center
            amrex::Print() << std::string(padding, ' ') << line << "\n";
        }
    }

    // Print bottom border (80 #)
    amrex::Print() << std::string(width, '#') << "\n";

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
  Fnum_int_f.resize(_mesh->L);
  Fnum_int_c.resize(_mesh->L);


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
  quadrule->xi_ref_quad_s_bdm.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (quadrule->qMp_s_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM)));    
  quadrule->xi_ref_quad_s_bdp.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (quadrule->qMp_s_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM)));    
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

  quadmat_bd.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Vector<amrex::Real>>(basefunc->Np_s,
                amrex::Vector<amrex::Real>(quadrule->qMp_s_bd)));  

  //Initialize quadrature weights for cell faces st quadratures 
  quad_weights_st_bd.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Real>(quadrule->qMp_st_bd));  

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
  
  //Set-up mesh interpolation
  amr_interpolator = std::make_shared<L2ProjInterp>();

  amr_interpolator->setNumericalMethod(shared_from_this());

  amr_interpolator->interp_proj_mat();

  amr_interpolator->flux_proj_mat();
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
  Fnum_int_f[lev].resize(AMREX_SPACEDIM);
  Fnum_int_c[lev].resize(AMREX_SPACEDIM);


  for(int d=0; d<AMREX_SPACEDIM; ++d){
    F[lev][d].resize(Q);
    Fm[lev][d].resize(Q);
    Fp[lev][d].resize(Q);
    DF[lev][d].resize(Q);
    DFm[lev][d].resize(Q);
    DFp[lev][d].resize(Q);
    Fnum[lev][d].resize(Q);
    Fnum_int_f[lev][d].resize(Q);
    Fnum_int_c[lev][d].resize(Q);

    H_p[lev][d].resize(Q);
    H_m[lev][d].resize(Q);
  }
  
  // Add verification that all data structures are properly sized
  AMREX_ASSERT(U_w[lev].size() == Q);
  AMREX_ASSERT(H_w[lev].size() == Q);
  AMREX_ASSERT(F[lev].size() == AMREX_SPACEDIM);
  for(int d = 0; d < AMREX_SPACEDIM; ++d) {
    AMREX_ASSERT(F[lev][d].size() == Q);
  }
  
  // Add MPI synchronization after data structure initialization
  amrex::ParallelDescriptor::Barrier();
}

//Init data for given level for specific solution component
void AmrDG::set_init_data_component(int lev,const BoxArray& ba,
                                    const DistributionMapping& dm, int q)
{ 
  auto _mesh = mesh.lock();

  // Add safety checks for data structure sizes
  AMREX_ASSERT(lev < H_w.size());
  AMREX_ASSERT(lev < U_w.size());
  AMREX_ASSERT(lev < F.size());
  AMREX_ASSERT(q < H_w[lev].size());
  AMREX_ASSERT(q < U_w[lev].size());

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
    // Add safety checks for F arrays
    AMREX_ASSERT(d < F[lev].size());
    AMREX_ASSERT(q < F[lev][d].size());
    
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

    Fnum_int_f[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,basefunc->Np_s,0);    
    Fnum_int_f[lev][d][q].setVal(0.0);    

    Fnum_int_c[lev][d][q].define(convert(ba, IntVect::TheDimensionVector(d)), dm,basefunc->Np_s,0);    
    Fnum_int_c[lev][d][q].setVal(0.0);    
  }
  
  // Add verification that all data structures are properly defined
  AMREX_ASSERT(H_w[lev][q].isDefined());
  AMREX_ASSERT(U_w[lev][q].isDefined());
  for(int d = 0; d < AMREX_SPACEDIM; ++d) {
    AMREX_ASSERT(F[lev][d][q].isDefined());
  }
}

void AmrDG::get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>* U_ptr,
                          amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                          const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //Can evalaute U_w (Np_s) at boundary using bd quadrature pts qM_s_bd or 
  //for the entire cell using qM_s
  //M=quadrule->qM_s;
  //N=basefunc->qM_s

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
              
          fab_u[q] = &(state_u[q]->get(mfi));
          u[q] = fab_u[q]->array();
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

          fab_u[q] = &(state_u[q]->get(mfi));
          u[q] = fab_u[q]->array();
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
        fab_h_w[q] = &(state_h_w[q]->get(mfi));

        uw[q] = fab_u_w[q]->const_array();
        hw[q] = fab_h_w[q]->array();
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
 
  auto _mesh = mesh.lock();

  amrex::Real dvol = _mesh->get_dvol(lev,d);
  // Time: [-1, 1] -> [0, Dt]   => Factor: Dt / 2.0
  // Space: [-1, 1]^(D-1) -> Face Area => Factor: dvol / 2^(D-1)
  amrex::Real jacobian = (Dt / 2.0) * (dvol / std::pow(2.0, AMREX_SPACEDIM - 1));

  //computes the numerical flux at the plus interface of a cell, i.e at idx i+1/2 
  amrex::Vector<amrex::MultiFab *> state_fnum(Q); 
  amrex::Vector<amrex::MultiFab *> state_fnum_int_f(Q); 
  amrex::Vector<amrex::MultiFab *> state_fnum_int_c(Q); 
  
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

    //TODO: should be passed as argument?
    state_fnum[q] = &(Fnum[lev][d][q]); 

    state_fnum_int_f[q] = &(Fnum_int_f[lev][d][q]);  
    state_fnum_int_c[q] = &(Fnum_int_c[lev][d][q]);
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector<amrex::FArrayBox *> fab_fnum(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnum(Q);

    amrex::Vector<amrex::FArrayBox *> fab_fnum_int_f(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnum_int_f(Q);

    amrex::Vector<amrex::FArrayBox *> fab_fnum_int_c(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnum_int_c(Q);

    
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
    for (MFIter mfi(*(state_fnum[0]), MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
  #else
      for (MFIter mfi(*(state_fnum[0]), true); mfi.isValid(); ++mfi)
  #endif
    {
      //Consider cell centered (valid) box [0,N] in each direction.
      //MFiter defined over face centered MultiFab with valid box
      //[0,N+1] in each direction (Face centered box has no ghost cells)
      const amrex::Box& _bx = mfi.tilebox();

      //Grow the box on the lower side to be able to access the left ghost cell
      //This make the box become [-1,N+1] in each direction
      amrex::Box bx = amrex::growLo(_bx, d, 1);

      //Get the LOCAL box limits for this specific rank/tile
      const amrex::IntVect& lo_idx = bx.smallEnd();
      
      for(int q=0 ; q<Q; ++q){
        fab_fnum[q]=&(state_fnum[q]->get(mfi));
        fab_fnum_int_f[q]=&(state_fnum_int_f[q]->get(mfi));
        fab_fnum_int_c[q]=&(state_fnum_int_c[q]->get(mfi));
  
        fab_fm[q] = state_fm[q]->fabPtr(mfi);
        fab_fp[q] = state_fp[q]->fabPtr(mfi);
        fab_dfm[q] = state_dfm[q]->fabPtr(mfi);
        fab_dfp[q] = state_dfp[q]->fabPtr(mfi);
        fab_um[q] = state_um[q]->fabPtr(mfi);
        fab_up[q] = state_up[q]->fabPtr(mfi);
        
        fnum[q]=fab_fnum[q]->array();
        fnum_int_f[q]=fab_fnum_int_f[q]->array();
        fnum_int_c[q]=fab_fnum_int_c[q]->array();

        fm[q] = fab_fm[q]->const_array();
        fp[q] = fab_fp[q]->const_array();
        dfm[q] = fab_dfm[q]->const_array();
        dfp[q] = fab_dfp[q]->const_array();
        um[q] = fab_um[q]->const_array();
        up[q] = fab_up[q]->const_array();
      }
            
      //compute the pointwise evaluations of the numerical flux
      amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
      {
        amrex::Array<int, AMREX_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
        if(idx[d] == lo_idx[d]){ return;}

        //check which indices it iterate across, i.e if last one is reachd
        for(int q=0 ; q<Q; ++q){
          fnum[q](i,j,k,m) = LLF_numflux(d,m,i,j,k,up[q],um[q],fp[q],fm[q],dfp[q],dfm[q]);  
        }
      }); 

      //compute the b- faces integral evaluations of the numerical flux
      amrex::ParallelFor(bx, N,[&] (int i, int j, int k, int n) noexcept
      {    
        amrex::Array<int, AMREX_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
        if(idx[d] == lo_idx[d]){ return;}

        // We are at interface i-1/2.
        // b = 1  => Coarse cell is at i-1 (Fine is at i). Face is HIGH/Plus side of Coarse.
        // b = -1 => Coarse cell is at i (Fine is at i-1). Face is LOW/Minus side of Coarse.
        int b = 0; 
        amrex::IntVect iv_left{AMREX_D_DECL(i,j,k)};
        iv_left[d] -= 1; // Cell i-1
        amrex::IntVect iv_right{AMREX_D_DECL(i,j,k)}; // Cell i

        // Logic: If one neighbor is NOT in the current level's valid box, 
        // but the other IS, we are at a coarse-fine interface.
        bool left_is_fine  = bx.contains(iv_left);
        bool right_is_fine = bx.contains(iv_right);

        if (!left_is_fine && right_is_fine) b = 1;   // Coarse on left
        else if (left_is_fine && !right_is_fine) b = -1; // Coarse on right

        //This cell might be a coarse neighbor.
        if (lev < _mesh->get_finest_lev()) {
            for(int q=0 ; q<Q; ++q){
                fnum_int_c[q](i,j,k,n) = 0.0;

                // Determine if face is Low or High relative to the coarse cell
                // If side == 1, face is HIGH side of cell i-1
                // If side == -1, face is LOW side of cell i
                
                for(int m=0; m<M; ++m){ 
                    if (b == 1) {
                        // High face projection
                        fnum_int_c[q](i,j,k,n) += fnum[q](i,j,k,m) * Mkbdp[d][n][m];
                    } else {
                        // Low face (default) projection
                        fnum_int_c[q](i,j,k,n) += fnum[q](i,j,k,m) * Mkbdm[d][n][m];
                    }
                }
                fnum_int_c[q](i,j,k,n) *= jacobian;
            }
        }

        //This cell might be a fine neighbor.
        if (lev > 0) {
          // Calculate Child Index
          int child_idx = 0;
          int bit_pos = 0;
          for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
              if (dir == d) continue;
              //Assume ref ratio of 2
              if (idx[dir] % 2 != 0) child_idx |= (1 << bit_pos);
              bit_pos++;
          }

          // Select matrix based on which side the coarse cell is!
          // If side == 0, this isn't a coarse-fine boundary, but we fill it anyway
          // for safety or internal level synchronization.
          int side_param = (b == 0) ? -1 : b; 
          const auto& P = amr_interpolator->get_flux_proj_mat(d, child_idx, side_param);

          for(int q=0 ; q<Q; ++q) {
              fnum_int_f[q](i,j,k,n) = 0.0;
              for(int m=0; m<M; ++m) { 
                  fnum_int_f[q](i,j,k,n) += P(n, m) * fnum[q](i,j,k,m);
              }
              fnum_int_f[q](i,j,k,n) *= jacobian;
          }
        }
      }); 
    }
  }
}

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

    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F[lev][d][q]); 
      state_fnum[d]=&(Fnum[lev][d][q]); 
    }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
    {
      amrex::Vector<const amrex::FArrayBox *> fab_f(AMREX_SPACEDIM);
      amrex::Vector<const amrex::FArrayBox *> fab_fnum(AMREX_SPACEDIM);

      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);   
      amrex::Vector<amrex::Array4<const amrex::Real> > fnum(AMREX_SPACEDIM); 

      
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
          
          f[d]= fab_f[d]->const_array();
          fnum[d]= fab_fnum[d]->const_array();
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
  
  // Add MPI synchronization after updating U_w for all components
  amrex::ParallelDescriptor::Barrier();
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
  
  // Add MPI synchronization after updating H_w for all components
  amrex::ParallelDescriptor::Barrier();
}

///////////////////////////////////////////////////////////////////////////


/*
class AmrDG : public amrex::AmrCore, public NumericalMethod
{
  public: 

    AmrDG(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, int _coord, 
          Vector<IntVect> const& _ref_ratios, Array<int,AMREX_SPACEDIM> const& _is_per,
          amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
          amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi, 
          amrex::Vector<amrex::Vector<int>> _bc_lo_type,
          amrex::Vector<amrex::Vector<int>> _bc_hi_type, amrex::Real _T,
          amrex::Real _CFL, int _p,
          int _t_regrid, int _t_outplt//, 
          //std::string _limiter_type, amrex::Real _TVB_M, 
          //amrex::Vector<amrex::Real> _AMR_TVB_C,
          //amrex::Vector<amrex::Real> _AMR_curl_C, 
          //amrex::Vector<amrex::Real> _AMR_div_C, 
          //amrex::Vector<amrex::Real> _AMR_grad_C, 
          //amrex::Vector<amrex::Real> _AMR_sec_der_C,
          //amrex::Real _AMR_sec_der_indicator, amrex::Vector<amrex::Real> _AMR_C
          , int _t_limit);

///////////////////////////////////////////////////////////////////////////
    //DG 
    //Initial Conditions, and level initialization
    void InitialCondition(int lev);
    
    amrex::Real Initial_Condition_U_w(int lev,int q,int n,int i,int j,int k) const;
    
    amrex::Real Initial_Condition_U(int lev,int q,int i,int j,int k,
                                    amrex::Vector<amrex::Real> xi) const;
 
      //Modal expansion
    void get_U_from_U_w(int c, amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                        amrex::Vector<amrex::MultiFab>* U_ptr,
                        amrex::Vector<amrex::Real> xi, bool is_predictor);
                                                          
    void get_u_from_u_w(int c, int i, int j, int k,
                        amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                        amrex::Vector<amrex::Array4< amrex::Real>>* u ,
                        amrex::Vector<amrex::Real> xi); 
                        
    void get_u_from_u_w(int c, int i, int j, int k,
                        amrex::Vector<amrex::Array4< amrex::Real>>* uw, 
                        amrex::Vector<amrex::Array4< amrex::Real>>* u ,
                        amrex::Vector<amrex::Real> xi);      
  
                                    
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
//LIMTIING/TAGGING

    amrex::Real minmodB(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                        bool &troubled_flag, int l) const;
    
    amrex::Real minmod(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                        bool &troubled_flag) const;  


    void Limiter_w(int lev); //, amrex::TagBoxArray& tags, char tagval
    
    void Limiter_linear_tvb(int i, int j, int k, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* uw, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* um,
                              amrex::Vector<amrex::Array4<amrex::Real>>* up, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* vw,
                              //amrex::Vector<amrex::Array4<amrex::Real>>* um_cpy,
                              int lev);  

    void AMRIndicator_tvb(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                          amrex::Vector<amrex::Array4<amrex::Real>>* um,
                          amrex::Vector<amrex::Array4<amrex::Real>>* up,
                          int l, amrex::Array4<char> const& tag,char tagval, 
                          bool& any_trouble); 
                          
    void AMRIndicator_second_derivative(int i, int j, int k, 
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                                        int l,amrex::Array4<char> const& tag,
                                        char tagval, bool& any_trouble);    
                                        
    void AMRIndicator_curl(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                          amrex::Array4<amrex::Real> const & curl_indicator,int l, 
                          bool flag_local,amrex::Array4<char> const& tag,
                          char tagval,bool& any_trouble);
                          
    void AMRIndicator_div(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                          amrex::Array4<amrex::Real> const & div_indicator,int l, 
                          bool flag_local,amrex::Array4<char> const& tag,
                          char tagval,bool& any_trouble);
                          
    void AMRIndicator_grad(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const  amrex::Real>>* uw, 
                          amrex::Array4<amrex::Real> const & grad_indicator,int l, 
                          bool flag_local,amrex::Array4<char> const& tag,
                          char tagval,bool& any_trouble);


    
    //Model Equation/Simulation settings and variables    
    
    int t_limit;
    std::string limiter_type;


    amrex::Vector<amrex::Real> AMR_C;
    
    amrex::Real AMR_curl_indicator;
    amrex::Vector<amrex::Real> AMR_curl_C;
    amrex::Vector<amrex::MultiFab> idc_curl_K;

    amrex::Real AMR_div_indicator;
    amrex::Vector<amrex::Real> AMR_div_C;
    amrex::Vector<amrex::MultiFab> idc_div_K;
    
    amrex::Real AMR_grad_indicator;
    amrex::Vector<amrex::Real> AMR_grad_C;
    amrex::Vector<amrex::MultiFab> idc_grad_K;    
    
    amrex::Real AMR_sec_der_indicator;
    amrex::Vector<amrex::Real> AMR_sec_der_C;
    
    amrex::Real TVB_M;
    amrex::Vector<amrex::Real> AMR_TVB_C;
///////////////////////////////////////////////////////////////////////////
};
*/
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
*/
