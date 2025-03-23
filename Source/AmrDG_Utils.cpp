#include "AmrDG.h"




/*
void AmrDG::NormDG()
{
  int _p = 1;
  int N = qMp_1d;
  int lev = 0;
  amrex::Vector<const amrex::MultiFab *> state_u_h(Q); 
  for(int q=0; q<Q;++q){state_u_h[q] = &(U_w[lev][q]);}
  
  amrex::BoxArray c_ba = U_w[lev][0].boxArray();
  
  //number of valid cells and respective volume
  int N_full =(int)(c_ba.numPts());           
  auto dx= geom[0].CellSizeArray();  
  amrex::Real vol = 0.0;
  #if (AMREX_SPACEDIM == 1)
    vol = dx[0];
  #elif (AMREX_SPACEDIM == 2)
    vol = dx[0]*dx[1];
  #elif (AMREX_SPACEDIM == 3)
    vol = dx[0]*dx[1]*dx[2];
  #endif
  amrex::Real V_level=(amrex::Real)(vol*(amrex::Real)(N_full));
  
  
  amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_full_tmp;
  Lpnorm_full_tmp.resize(Q);    

#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector<const amrex::FArrayBox *> fab_u_h(Q);
    amrex::Vector< amrex::Array4<const amrex::Real>> uh(Q);  
    
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_u_h[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_u_h[0]),true); mfi.isValid(); ++mfi)
    #endif    
    {
      const amrex::Box& bx_tmp = mfi.tilebox();
      //const amrex::Box& bx_tmp = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        fab_u_h[q] = state_u_h[q]->fabPtr(mfi);
        uh[q] = fab_u_h[q]->const_array();
      }       
    
      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
        {           
          amrex::Real cell_Lpnorm =0.0;
          amrex::Real w;
          amrex::Real f;
          
          //quadrature
          for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){ 
            //quad weights for each quadrature point
            w = 1.0;
            for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
              w*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad_L2proj[m][d_]),2);
            }
            
            amrex::Real u_h = 0.0;  

            for (int n = 0; n < Np; ++n){  
              u_h+=uh[q](i,j,k,n)*Phi(n, xi_ref_GLquad_L2proj[m]);
            }
        
            amrex::Real u = 0.0; 
            u =  Initial_Condition_U(lev,q,i,j,k,xi_ref_GLquad_L2proj[m]);
            f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
            cell_Lpnorm += (f*w);
          }
            
          amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
          #pragma omp critical
          {
            Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);    
          }      
        });  
      }          
    }
    for(int q=0 ; q<Q; ++q){
      amrex::Real global_Lpnorm = 0.0;
      global_Lpnorm = std::accumulate(Lpnorm_full_tmp[q].begin(), 
                                       Lpnorm_full_tmp[q].end(), 0.0);
                                                    
      ParallelDescriptor::ReduceRealSum(global_Lpnorm);//sum up all the ranks level norms
   
      amrex::Real Lpnorm=std::pow(global_Lpnorm/V_level, 1.0/(amrex::Real)_p);
      
      Print().SetPrecision(17)<<"--level "<<lev<<"--"<<"\n";
      Print().SetPrecision(17)<< "L"<<_p<<" error norm:  "<<Lpnorm<<" | "<<
                          "DG Order:  "<<p+1<<" | solution component: "<<q<<"\n"; 
    }
  }
}  

void AmrDG::Conservation(int lev, int M, amrex::Vector<amrex::Vector<amrex::Real>> xi, int d)
{
  //function to be called before and after timesteps,limiters are applied etc to check
  //that the quantities of interest are conserved
  //Should compute a sum over the domain and print it out
  amrex::Vector<const amrex::MultiFab *> state_u_w(Q);   
  amrex::Vector<amrex::MultiFab *> state_u(Q); 

  //evlauate polynomial at location of interest
  //might not be used depending on model equation implementation
  for(int m = 0; m<M ; ++m){
    get_U_from_U_w(m,&U_w[lev], &U[lev], xi[m],false);
  }
  
  for(int q=0; q<Q; ++q){
    state_u_w[q]=&(U_w[lev][q]);
    state_u[q]=&(U[lev][q]);   
  } 

  amrex::Vector<amrex::Vector<amrex::Real>> conserved_rank;
  conserved_rank.resize(Q);
  
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
      const amrex::Box& bx = mfi.tilebox();
      for(int q=0 ; q<Q; ++q){
        fab_u_w[q] = state_u_w[q]->fabPtr(mfi);
        uw[q] = fab_u_w[q]->const_array();
             
        fab_u[q]=&((*(state_u[q]))[mfi]);
        u[q]=(*(fab_u[q])).array();
      } 
      
      for(int q=0 ; q<Q; ++q){ 
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {  
          amrex::Real cons = sim->model_pde->pde_conservation(lev,d,q,i,j,k,&uw,&u);
          #pragma omp critical
          {
            conserved_rank[q].push_back(cons);
          }       
        }); 
      }
    }
  }
  
  for(int q=0 ; q<Q; ++q){
    amrex::Real cons_glob=0.0;
    if (!conserved_rank[q].empty()) {
      cons_glob =std::accumulate(conserved_rank[q].begin(), conserved_rank[q].end(), 0.0);
    }    
    ParallelDescriptor::ReduceRealSum(cons_glob);
    Print().SetPrecision(17)<< "Conservation solution component :"<<q<<" | "<<cons_glob<<"\n"; 
  }    
}

void AmrDG::DEBUG_print_MFab() 
{ 
  //this function is used for debugging and prints out the specified MFab 
  //and if wanted also multilevel data. Is jsut a cleaner option
  //than copy paste the loop in the already dense code
  //user should implement wathever they want
  int q = 2;
  int lev = 0;
  int dim = 0;

  //amrex::MultiFab& state_c = U_w[lev][q];
  //amrex::MultiFab& state_c = H_w[lev][q];
  //amrex::MultiFab& state_c = Fnum[lev][dim][q];
  amrex::MultiFab& state_c = Fnumm_int[lev][dim][q];
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
          for(int n = 0; n<Np; ++n) {
            Print(3) <<i<<","<<j<<"  | w="<<n<<"| "<<uc(i,j,k,n)<<"\n";
          }       
        } 
      }       
    }
  }
  
  Print() <<"        "<<"\n";
}

*/