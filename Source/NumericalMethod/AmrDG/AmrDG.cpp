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
  //amrex::MultiFab& state_c = F[idx_ldq(lev,dim,q)];
  //amrex::MultiFab& state_c = U_w(lev,q);
  //amrex::MultiFab& state_c = H_w(lev,q);
  //amrex::MultiFab& state_c = Fnum[idx_ldq(lev,dim,q)];
  amrex::MultiFab& state_c = DFp(lev,q);
  //amrex::MultiFab& state_c = H_p(lev,q);

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

void AmrDG::settings(int _p, amrex::Real _T, amrex::Real _c_dt) {
  p = _p;
  T = _T;
  c_dt = _c_dt;
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
{ 
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
  
  //Set flat vectors size (L*Q, L*D*Q, L*D, or L)
  U_w.resize(_mesh->L, Q);

  U.resize(_mesh->L, Q);

  if(flag_source_term){S.resize(_mesh->L, Q);}

  U_center.resize(_mesh->L, Q);

  F.resize(_mesh->L, AMREX_SPACEDIM, Q);

  Fm.resize(_mesh->L, Q);

  Fp.resize(_mesh->L, Q);

  DFm.resize(_mesh->L, Q);

  DFp.resize(_mesh->L, Q);

  Fnum.resize(_mesh->L, AMREX_SPACEDIM, Q);

  Fnum_int_f.resize(_mesh->L, AMREX_SPACEDIM, Q);

  Fnum_int_c.resize(_mesh->L, AMREX_SPACEDIM, Q);

  cf_face_b_coarse.resize(_mesh->L, AMREX_SPACEDIM);
  cf_face_b_fine.resize(_mesh->L, AMREX_SPACEDIM);
  cf_face_child_idx.resize(_mesh->L, AMREX_SPACEDIM);

  H_w.resize(_mesh->L, Q);

  H.resize(_mesh->L, Q);

  rhs_corr.resize(_mesh->L);

  rhs_pred.resize(_mesh->L);

  H_p.resize(_mesh->L, Q);

  H_m.resize(_mesh->L, Q);

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
                              
  //Initialize generalized Vandermonde matrix and inverse
  V = Eigen::MatrixXd::Zero(quadrule->qMp_st, basefunc->Np_st);
  Vinv = Eigen::MatrixXd::Zero(basefunc->Np_st, quadrule->qMp_st);

  //Initialize L2 projection quadrature matrix
  quadmat = Eigen::MatrixXd::Zero(basefunc->Np_s, quadrule->qMp_s);

  quadmat_bd.resize(AMREX_SPACEDIM);
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    quadmat_bd[d] = Eigen::MatrixXd::Zero(basefunc->Np_s, quadrule->qMp_s_bd);
  }

  //Initialize quadrature weights for cell faces st quadratures
  quad_weights_st_bdm.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Real>(quadrule->qMp_st_bd));
  quad_weights_st_bdp.resize(AMREX_SPACEDIM,amrex::Vector<amrex::Real>(quadrule->qMp_st_bd));

  //Initialize generalized Element matrices for ADER-DG corrector
  Mk_corr = Eigen::MatrixXd::Zero(basefunc->Np_s, basefunc->Np_s);
  Mk_corr_src = Eigen::MatrixXd::Zero(basefunc->Np_s, quadrule->qMp_st);

  Sk_corr.resize(AMREX_SPACEDIM);
  Mkbdm.resize(AMREX_SPACEDIM);
  Mkbdp.resize(AMREX_SPACEDIM);
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    Sk_corr[d] = Eigen::MatrixXd::Zero(basefunc->Np_s, quadrule->qMp_st);
    Mkbdm[d] = Eigen::MatrixXd::Zero(basefunc->Np_s, quadrule->qMp_st_bd);
    Mkbdp[d] = Eigen::MatrixXd::Zero(basefunc->Np_s, quadrule->qMp_st_bd);
  }

  //Initialize generalized Element matrices for ADER predictor
  Mk_h_w = Eigen::MatrixXd::Zero(basefunc->Np_st, basefunc->Np_st);
  Mk_h_w_inv = Eigen::MatrixXd::Zero(basefunc->Np_st, basefunc->Np_st);
  Mk_pred = Eigen::MatrixXd::Zero(basefunc->Np_st, basefunc->Np_s);
  Mk_pred_src = Eigen::MatrixXd::Zero(basefunc->Np_st, basefunc->Np_st);
  Mk_pred_srcVinv = Eigen::MatrixXd::Zero(basefunc->Np_st, quadrule->qMp_st);

  Sk_pred.resize(AMREX_SPACEDIM);
  Sk_predVinv.resize(AMREX_SPACEDIM);
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    Sk_pred[d] = Eigen::MatrixXd::Zero(basefunc->Np_st, basefunc->Np_st);
    Sk_predVinv[d] = Eigen::MatrixXd::Zero(basefunc->Np_st, quadrule->qMp_st);
  }
    
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
    sum+= quadmat(dcomp +comp,m)*bc[m];
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
  //Flat vectors already pre-sized in init(); just define rhs temporaries here
  auto _mesh = mesh.lock();

  rhs_corr[lev].define(ba, dm, basefunc->Np_s, _mesh->nghost);
  rhs_pred[lev].define(ba, dm, basefunc->Np_st, _mesh->nghost);

  // Add MPI synchronization after data structure initialization
  amrex::ParallelDescriptor::Barrier();
}

//Init data for given level for specific solution component
void AmrDG::set_init_data_component(int lev,const BoxArray& ba,
                                    const DistributionMapping& dm, int q)
{
  auto _mesh = mesh.lock();

  H_w(lev,q).define(ba, dm, basefunc->Np_st, _mesh->nghost);
  H_w(lev,q).setVal(0.0);

  H(lev,q).define(ba, dm, quadrule->qMp_st, _mesh->nghost);
  H(lev,q).setVal(0.0);

  U_w(lev,q).define(ba, dm, basefunc->Np_s, _mesh->nghost);
  U_w(lev,q).setVal(0.0);

  U(lev,q).define(ba, dm, quadrule->qMp_st, _mesh->nghost);
  U(lev,q).setVal(0.0);

  U_center(lev,q).define(ba, dm, 1, _mesh->nghost);
  U_center(lev,q).setVal(0.0);

  if(flag_source_term){S(lev,q).define(ba, dm, quadrule->qMp_st, _mesh->nghost);
  S(lev,q).setVal(0.0);}

  H_p(lev,q).define(ba, dm, quadrule->qMp_st_bd, _mesh->nghost);
  H_m(lev,q).define(ba, dm, quadrule->qMp_st_bd, _mesh->nghost);
  Fm(lev,q).define(ba, dm, quadrule->qMp_st_bd, _mesh->nghost);
  Fp(lev,q).define(ba, dm, quadrule->qMp_st_bd, _mesh->nghost);
  DFm(lev,q).define(ba, dm, quadrule->qMp_st_bd, _mesh->nghost);
  DFp(lev,q).define(ba, dm, quadrule->qMp_st_bd, _mesh->nghost);

  for(int d=0; d<AMREX_SPACEDIM; ++d){
    F(lev,d,q).define(ba, dm, quadrule->qMp_st, _mesh->nghost);

    Fnum(lev,d,q).define(convert(ba, IntVect::TheDimensionVector(d)), dm, quadrule->qMp_st_bd, 0);

    Fnum_int_f(lev,d,q).define(convert(ba, IntVect::TheDimensionVector(d)), dm, basefunc->Np_s, 0);

    Fnum_int_c(lev,d,q).define(convert(ba, IntVect::TheDimensionVector(d)), dm, basefunc->Np_s, 0);
  }
}

void AmrDG::get_U_from_U_w(int M, int N, amrex::MultiFab* _U,
                          amrex::MultiFab* _U_w,
                          const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //Can evalaute U_w (Np_s) at boundary using bd quadrature pts qM_s_bd or
  //for the entire cell using qM_s
  //M=quadrule->qM_s;
  //N=basefunc->qM_s

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
      amrex::Vector<amrex::Array4<const amrex::Real>> uw(Q);
      amrex::Vector<amrex::Array4<amrex::Real>> u(Q);

      #ifdef AMREX_USE_OMP
      for (MFIter mfi(_U_w[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
      #else
      for (MFIter mfi(_U_w[0],true); mfi.isValid(); ++mfi)
      #endif
      {
        const amrex::Box& bx = mfi.growntilebox();

        for(int q=0 ; q<Q; ++q){
          uw[q] = _U_w[q].const_array(mfi);
          u[q] = _U[q].array(mfi);
        }

        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          for(int q=0 ; q<Q; ++q){
            for(int m = 0; m<M ; ++m){
              (u[q])(i,j,k,m)=0.0;
            }
          }
        });

        amrex::ParallelFor(bx,N,[&] (int i, int j, int k,int n) noexcept
        {
          for(int q=0 ; q<Q; ++q){
            for(int m = 0; m<M ; ++m){
              (u[q])(i,j,k,m)+=((uw[q])(i,j,k,n)*basefunc->phi_s(n,basefunc->basis_idx_s,xi[m]));
            }
          }
        });
      }
    }
}

void AmrDG::get_H_from_H_w(int M, int N, amrex::MultiFab* _H,
                          amrex::MultiFab* _H_w,
                          const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //Can evalaute H_w (Np_st) at boundary using bd quadrature pts qM_st_bd or
  //for the entire cell using qM_st

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
      amrex::Vector<amrex::Array4<const amrex::Real>> hw(Q);
      amrex::Vector<amrex::Array4<amrex::Real>> h(Q);

      #ifdef AMREX_USE_OMP
      for (MFIter mfi(_H_w[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
      #else
      for (MFIter mfi(_H_w[0],true); mfi.isValid(); ++mfi)
      #endif
      {
        const amrex::Box& bx = mfi.growntilebox();

        for(int q=0 ; q<Q; ++q){
          hw[q] = _H_w[q].const_array(mfi);
          h[q] = _H[q].array(mfi);
        }
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          for(int q=0 ; q<Q; ++q){
            for(int m = 0; m<M ; ++m){
              (h[q])(i,j,k,m)=0.0;
            }
          }
        });

        amrex::ParallelFor(bx,N,[&] (int i, int j, int k,int n) noexcept
        {
          for(int q=0 ; q<Q; ++q){
            for(int m = 0; m<M ; ++m){
              (h[q])(i,j,k,m)+=((hw[q])(i,j,k,n)*basefunc->phi_st(n,basefunc->basis_idx_st,xi[m]));
            }
          }
        });
      }
    }
}

void AmrDG::set_predictor(const amrex::MultiFab* _U_w,
                          amrex::MultiFab* _H_w)
{

  #ifdef AMREX_USE_OMP
  #pragma omp parallel
  #endif
  {
    amrex::Vector<amrex::Array4<amrex::Real>> hw(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> uw(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(_H_w[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(_H_w[0],true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        uw[q] = _U_w[q].const_array(mfi);
        hw[q] = _H_w[q].array(mfi);
      }

      amrex::ParallelFor(bx, basefunc->Np_st,[&] (int i, int j, int k, int n) noexcept
      {
        for(int q=0 ; q<Q; ++q){
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
        }
      });      
    }  
  }  
}

void AmrDG::numflux(int lev,int d,int M, int N,
                    amrex::MultiFab* _U_m,
                    amrex::MultiFab* _U_p,
                    amrex::MultiFab* _F_m,
                    amrex::MultiFab* _F_p,
                    amrex::MultiFab* _DF_m,
                    amrex::MultiFab* _DF_p)
{

  auto _mesh = mesh.lock();

  amrex::Real dvol = _mesh->get_dvol(lev,d);
  amrex::Real jacobian = (Dt / 2.0) * (dvol / std::pow(2.0, AMREX_SPACEDIM - 1));

  for (int q = 0; q < Q; ++q) {
    Fnum_int_f(lev,d,q).setVal(0.0);
    Fnum_int_c(lev,d,q).setVal(0.0);
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  {
    amrex::Vector<amrex::Array4<amrex::Real>> fnum(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> fnum_int_f(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> fnum_int_c(Q);

    amrex::Vector<amrex::Array4<const amrex::Real>> fm(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> fp(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> dfm(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> dfp(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> um(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> up(Q);

  #ifdef AMREX_USE_OMP
    for (MFIter mfi(Fnum(lev,d,0), MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
  #else
      for (MFIter mfi(Fnum(lev,d,0), true); mfi.isValid(); ++mfi)
  #endif
    {

      const amrex::Box& bx = mfi.tilebox();

      for(int q=0 ; q<Q; ++q){
        fnum[q] = Fnum(lev,d,q).array(mfi);
        fnum_int_f[q] = Fnum_int_f(lev,d,q).array(mfi);
        fnum_int_c[q] = Fnum_int_c(lev,d,q).array(mfi);

        fm[q] = _F_m[q].const_array(mfi);
        fp[q] = _F_p[q].const_array(mfi);
        dfm[q] = _DF_m[q].const_array(mfi);
        dfp[q] = _DF_p[q].const_array(mfi);
        um[q] = _U_m[q].const_array(mfi);
        up[q] = _U_p[q].const_array(mfi);
      }

      amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
      {
        for(int q=0 ; q<Q; ++q){
          fnum[q](i,j,k,m) = LLF_numflux(d,m,i,j,k,up[q],um[q],fp[q],fm[q],dfp[q],dfm[q]);
        }
      });


      if ((_mesh->L > 1))
      {
        auto const& bc_arr = cf_face_b_coarse(lev,d).const_array(mfi);
        auto const& bf_arr = cf_face_b_fine(lev,d).const_array(mfi);
        auto const& ci_arr = cf_face_child_idx(lev,d).const_array(mfi);

        amrex::ParallelFor(bx, N,[&] (int i, int j, int k, int n) noexcept
        {
          if (lev < _mesh->get_finest_lev())
          {
              int b = bc_arr(i,j,k);
              if (b != 0)
              {
                  auto& mat = (b == 1) ? Mkbdp[d] : Mkbdm[d];

                  for (int q = 0; q < Q; ++q) {
                    amrex::Real sum = 0.0;
                    for (int m = 0; m < M; ++m) {
                        sum += fnum[q](i,j,k,m) * mat(n,m);
                    }
                    fnum_int_c[q](i,j,k,n) = sum * jacobian;
                  }
              }
          }

          if (lev > 0)
          {
              int b = bf_arr(i,j,k);
              if (b != 0)
              {
                  int child_idx = ci_arr(i,j,k);
                  const auto& P = amr_interpolator->get_flux_proj_mat(d, child_idx, b);

                  for (int q = 0; q < Q; ++q) {
                      amrex::Real sum = 0.0;
                      for (int m = 0; m < M; ++m) {
                          sum += P(n, m) * fnum[q](i,j,k,m);
                      }
                      fnum_int_f[q](i,j,k,n) = sum * jacobian;
                  }
              }
          }
        });
      }
    }
  }
}

void AmrDG::update_U_w(int lev)
{
  auto _mesh = mesh.lock();

  const auto dx = _mesh->get_dx(lev);
  amrex::Real vol = _mesh->get_dvol(lev);

  for(int q=0; q<Q; ++q){
    amrex::MultiFab& state_u_w = U_w(lev,q);

    rhs_corr[lev].setVal(0.0);

    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM);
    amrex::Vector<const amrex::MultiFab *> state_fnum(AMREX_SPACEDIM);

    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F(lev,d,q));
      state_fnum[d]=&(Fnum(lev,d,q));
    }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
    {
      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);
      amrex::Vector<amrex::Array4<const amrex::Real> > fnum(AMREX_SPACEDIM);

      #ifdef AMREX_USE_OMP
      for (MFIter mfi(state_u_w,MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
      #else
      for (MFIter mfi(state_u_w,true); mfi.isValid(); ++mfi)
      #endif
      {
        const amrex::Box& bx = mfi.tilebox();

        amrex::Array4<amrex::Real> const& uw = state_u_w.array(mfi);
        amrex::Array4<amrex::Real> const& rhs = rhs_corr[lev].array(mfi);

        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          f[d] = state_f[d]->const_array(mfi);
          fnum[d] = state_fnum[d]->const_array(mfi);
        }

        amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
        {
          rhs(i,j,k,n)+=(Mk_corr(n,n)*uw(i,j,k,n));
        });

        int shift[] = {0,0,0};
        for  (int d = 0; d < AMREX_SPACEDIM; ++d){
          amrex::Real S_norm = (Dt/(amrex::Real)dx[d]);
          amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
          {
            for  (int m = 0; m < quadrule->qMp_st; ++m){
              rhs(i,j,k,n)+=S_norm*(Sk_corr[d](n,m)*((f)[d])(i,j,k,m));
            }
          });

          amrex::Real Mbd_norm = (Dt/(amrex::Real)dx[d]);
          shift[d] = 1;
          amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
          {
            for  (int m = 0; m < quadrule->qMp_st_bd; ++m){
              rhs(i,j,k,n)-=(Mbd_norm*(Mkbdp[d](n,m)*((fnum)[d])(i+shift[0],j+shift[1], k+shift[2],m)
                                      -Mkbdm[d](n,m)*((fnum)[d])(i,j,k,m)));
            }
          });
          shift[d] = 0;
        }

        if(flag_source_term)
        {
          amrex::Array4<amrex::Real> const& source = S(lev,q).array(mfi);
          amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
          {
            for  (int m = 0; m < quadrule->qMp_st; ++m){
              rhs(i,j,k,n)+=((Dt/2.0)*Mk_corr_src(n,m)*source(i,j,k,m));
            }
          });
        }

        amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
        {
          rhs(i,j,k,n)/=Mk_corr(n,n);
          uw(i,j,k,n) = rhs(i,j,k,n);
        });
      }
    }
  }
}

void AmrDG::update_H_w(int lev)
{
  auto _mesh = mesh.lock();

  const auto dx = _mesh->get_dx(lev);

  for(int q=0; q<Q; ++q)
  {
    amrex::MultiFab& state_h_w = H_w(lev,q);
    amrex::MultiFab& state_u_w = U_w(lev,q);

    amrex::Vector<const amrex::MultiFab *> state_f(AMREX_SPACEDIM);

    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      state_f[d]=&(F(lev,d,q));
    }

    rhs_pred[lev].setVal(0.0); 
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
    {
      amrex::Vector<amrex::Array4<const amrex::Real> > f(AMREX_SPACEDIM);

      #ifdef AMREX_USE_OMP
      for (MFIter mfi(state_h_w,MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
      #else
      for (MFIter mfi(state_h_w,true); mfi.isValid(); ++mfi)
      #endif
      {
        const amrex::Box& bx = mfi.growntilebox();

        amrex::Array4<amrex::Real> const& hw = state_h_w.array(mfi);
        amrex::Array4<amrex::Real> const& uw = state_u_w.array(mfi);
        amrex::Array4<amrex::Real> const& rhs = rhs_pred[lev].array(mfi);

        for(int d = 0; d < AMREX_SPACEDIM; ++d){
          f[d] = state_f[d]->const_array(mfi);
        } 
        
        amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
        {
          hw(i,j,k,n) = 0.0;
          for(int m =0; m<basefunc->Np_s; ++m)
          {
            rhs(i,j,k,n) += Mk_pred(n,m)*uw(i,j,k,m);
          }
        });

        for(int d=0; d<AMREX_SPACEDIM; ++d)
        {
          amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
          {
            for(int m =0; m<quadrule->qMp_st; ++m)
            {
              rhs(i,j,k,n) -= ((Dt/(amrex::Real)dx[d])*Sk_predVinv[d](n,m)*f[d](i,j,k,m));
            }
          });
        }

        if(flag_source_term){
          amrex::Array4<amrex::Real> const& source = S(lev,q).array(mfi);

          amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
          {
            for(int m =0; m<quadrule->qMp_st; ++m)
            {
              rhs(i,j,k,n)+=(Dt/2.0)*Mk_pred_srcVinv(n,m)*source(i,j,k,m);
            }
          });
        }

        amrex::ParallelFor(bx, basefunc->Np_st, [&] (int i, int j, int k, int n) noexcept
        {
          for(int m =0; m<basefunc->Np_st; ++m){
            hw(i,j,k,n) += Mk_h_w_inv(n,m)*rhs(i,j,k,m);
          }
        });
      }
    }
  }
}
