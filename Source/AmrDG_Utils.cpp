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
#include <AMReX_Interpolater.H>
#include <AMReX_Interp_C.H>
#include <AMReX_MFInterp_C.H>

#include <climits>

void AmrDG::LpNorm_DG_AMR(int _p, amrex::Vector<amrex::Vector<amrex::Real>> quad_pt, int N) const
{
  //for debugging purposes might be usefull to look at the norm on individual levels
  //without discarting the overlap
  //false== output norm of idividual levels
  //true== output global norm
  bool flag_no_AMR_overlap = false;
 
  if(flag_no_AMR_overlap)
  {
    amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_multilevel;
    Lpnorm_multilevel.resize(Q);
    amrex::Vector<amrex::Real> V_level;
  
    for(int l=0; l<=finest_level; ++l)
    {
      amrex::Vector<const amrex::MultiFab *> state_u_h(Q);   
      amrex::Vector<const amrex::FArrayBox *> fab_u_h(Q);
      amrex::Vector< amrex::Array4<const amrex::Real>> uh(Q);  
      
      amrex::Vector<amrex::MultiFab> U_h_DG;
      U_h_DG.resize(Q);
    
      amrex::Vector<amrex::MultiFab> U_h_DG_intrsct_f;
      U_h_DG_intrsct_f.resize(Q);
      amrex::Vector<amrex::MultiFab> U_h_DG_intrsct_c;
      U_h_DG_intrsct_c.resize(Q);
      
      for(int q=0; q<Q;++q){
        amrex::BoxArray c_ba = U_w[l][q].boxArray();    
        U_h_DG[q].define(c_ba, U_w[l][q].DistributionMap(), Np, nghost);       
        amrex::MultiFab::Copy(U_h_DG[q], U_w[l][q], 0, 0, Np, nghost);        
      }
      
      //get number of cells of full level and intersection level
      amrex::BoxArray c_ba = U_w[l][0].boxArray();
      int N_full =(int)(c_ba.numPts()); 
      
      int N_overlap=0;
      if(l!=finest_level){
        amrex::BoxArray f_ba = U_w[l+1][0].boxArray();
        amrex::BoxArray f_ba_c = f_ba.coarsen(ref_ratio[l]); 
        N_overlap=(int)(f_ba_c.numPts()); 
      }
      
      auto dx= geom[l].CellSizeArray();  
      amrex::Real vol = 0.0;
      #if (AMREX_SPACEDIM == 1)
        vol = dx[0];
      #elif (AMREX_SPACEDIM == 2)
        vol = dx[0]*dx[1];
      #elif (AMREX_SPACEDIM == 3)
        vol = dx[0]*dx[1]*dx[2];
      #endif

      V_level.push_back((amrex::Real)(vol*(amrex::Real)(N_full-N_overlap)));

      //Compute Lp norm on full level
      for(int q=0; q<Q;++q){state_u_h[q] = &(U_h_DG[q]);}
      
      //vector to accumulate all the full level norm (reduction sum of all cells norms)
      amrex::Vector<amrex::Real> Lpnorm_full;
      Lpnorm_full.resize(Q);
      amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_full_tmp;
      Lpnorm_full_tmp.resize(Q);
      
      #ifdef AMREX_USE_OMP
      #pragma omp parallel 
      #endif
      { 
        for (MFIter mfi(*(state_u_h)[0],true); mfi.isValid(); ++mfi){           
          const amrex::Box& bx_tmp = mfi.tilebox();

          for(int q=0 ; q<Q; ++q){
            fab_u_h[q] = state_u_h[q]->fabPtr(mfi);
            uh[q] = fab_u_h[q]->const_array();
          } 
            
            
          if(l!=finest_level){

            amrex::BoxArray f_ba = U_w[l+1][0].boxArray();
            amrex::BoxArray ba_c = f_ba.coarsen(ref_ratio[l]); 
            const amrex::BoxList f_ba_lst(ba_c);
            //amrex::BoxList f_ba_lst_noover = removeOverlap(f_ba_lst);
            
            amrex::BoxList  f_ba_lst_compl = complementIn(bx_tmp,f_ba_lst);
                
            amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
            {
              bool flag_is_overlap = false;
              ///*
              for (const amrex::Box& bx : f_ba_lst_compl)
              { 
                amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                if(bx.contains(iv))
                {
                  flag_is_overlap=true;
                }            
              }
              //*/
              /*
              amrex::IntVect iv(AMREX_D_DECL(i, j, k));
              for (const amrex::Box& bx : f_ba_lst_noover) {
                if(bx.contains(iv))
                {
                  flag_is_overlap=true;
                  break;
                }  
              }
              */
              
              if(!flag_is_overlap)
              {
                for(int q=0 ; q<Q; ++q){
                  amrex::Real cell_Lpnorm =0.0;
                  amrex::Real w;
                  amrex::Real f;
                  
                  for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){ 
                    //quad weights for each quadrature point
                    w = 1.0;
                    for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
                      w*=2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
                    }
                    
                    amrex::Real u_h = 0.0;            
                    for (int n = 0; n < Np; ++n){  
                      u_h+=uh[q](i,j,k,n)*Phi(n, quad_pt[m]);
                    }
                        
                    amrex::Real u = 0.0; 
                    u = Initial_Condition_U(l,q,i,j,k,quad_pt[m]);
                                
                    f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
                    cell_Lpnorm += (f*w);
                    }
                    amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
                    #pragma omp critical
                    {
                      Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);    
                    }
                  }            
                }
            });                                
          }  
          else
          {
            amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
            {
              for(int q=0 ; q<Q; ++q){
                amrex::Real cell_Lpnorm =0.0;
                amrex::Real w;
                amrex::Real f;
                
                for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){ 
                  //quad weights for each quadrature point
                  w = 1.0;
                  for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
                    w*=2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
                  }
                  
                  amrex::Real u_h = 0.0;            
                  for (int n = 0; n < Np; ++n){  
                    u_h+=uh[q](i,j,k,n)*Phi(n, quad_pt[m]);
                  }
                      
                  amrex::Real u = 0.0; 
                  u = Initial_Condition_U(l,q,i,j,k,quad_pt[m]);
                              
                  f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
                  cell_Lpnorm += (f*w);
                  }
                  amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
                  #pragma omp critical
                  {
                    Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);    
                  }
                }           
            });        
          }   
        }
      }
      for(int q=0 ; q<Q; ++q){
        amrex::Real global_Lpnorm = 0.0;
        global_Lpnorm = std::accumulate(Lpnorm_full_tmp[q].begin(), 
                                         Lpnorm_full_tmp[q].end(), 0.0);
                                         //level norm for cells of this rank
                        
        ParallelDescriptor::ReduceRealSum(global_Lpnorm);//sum up all the ranks level norms
        Lpnorm_full[q] = global_Lpnorm;
      } 
        
      for(int q=0 ; q<Q; ++q){
        Lpnorm_multilevel[q].push_back((amrex::Real)Lpnorm_full[q]);        
      }      
    }
    
    amrex::Real V_amr = (amrex::Real)std::accumulate(V_level.begin(),V_level.end(), 0.0);  
    for(int q=0 ; q<Q; ++q){
      amrex::Real Lpnorm = std::accumulate(Lpnorm_multilevel[q].begin(), 
                                          Lpnorm_multilevel[q].end(), 0.0);
                                          
      Lpnorm=std::pow(Lpnorm/V_amr, 1.0/(amrex::Real)_p);
      Print().SetPrecision(17)<<"--multilevel--"<<"\n";
      Print().SetPrecision(17)<< "L"<<_p<<" error norm:  "<<Lpnorm<<" | "<<
                          "DG Order:  "<<p+1<<" | solution component: "<<q<<"\n"; 
    }  
  }
  else///////////////////////////////////////////////////////////////////////////
  {
    for(int l=0; l<=finest_level; ++l)
    {
      
      amrex::Vector<const amrex::MultiFab *> state_u_h(Q);   
      
      amrex::Vector<amrex::MultiFab> U_h_DG;
      U_h_DG.resize(Q);  
      
      for(int q=0; q<Q;++q){
        amrex::BoxArray c_ba = U_w[l][q].boxArray();    
        U_h_DG[q].define(c_ba, U_w[l][q].DistributionMap(), Np, nghost);       
        amrex::MultiFab::Copy(U_h_DG[q], U_w[l][q], 0, 0, Np, nghost);        
      }
      
      for(int q=0; q<Q;++q){state_u_h[q] = &(U_h_DG[q]);}
      
      amrex::BoxArray c_ba = U_w[l][0].boxArray();
      
      int N_full =(int)(c_ba.numPts());           
      auto dx= geom[l].CellSizeArray();  
      amrex::Real vol = 0.0;
      #if (AMREX_SPACEDIM == 1)
        vol = dx[0];
      #elif (AMREX_SPACEDIM == 2)
        vol = dx[0]*dx[1];
      #elif (AMREX_SPACEDIM == 3)
        vol = dx[0]*dx[1]*dx[2];
      #endif
      amrex::Real V_level=(amrex::Real)(vol*(amrex::Real)(N_full));


      //vector to accumulate all the full level norm (reduction sum of all cells norms)
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
                w*=2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
              }
              
              amrex::Real u_h = 0.0;  
              //u_h+=uh[q](i,j,k,0);
              ///*
              for (int n = 0; n < Np; ++n){  
                u_h+=uh[q](i,j,k,n)*Phi(n, quad_pt[m]);
              }
              //*/
              amrex::Real u = 0.0; 
              u =  Initial_Condition_U(l,q,i,j,k,quad_pt[m]);
              f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
              cell_Lpnorm += (f*w);
            }
              
            amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
            #pragma omp critical
            {
              Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);    
            }      
            //if(q==0){amrex::Real tmp = cell_Lpnorm*coeff;Print() <<i<<","<<j<<" | "<<tmp<<"\n";}
          });  
        }
      }
      for(int q=0 ; q<Q; ++q){
        amrex::Real global_Lpnorm = 0.0;
        global_Lpnorm = std::accumulate(Lpnorm_full_tmp[q].begin(), 
                                         Lpnorm_full_tmp[q].end(), 0.0);
                                                      
        ParallelDescriptor::ReduceRealSum(global_Lpnorm);//sum up all the ranks level norms
     
        amrex::Real Lpnorm=std::pow(global_Lpnorm/V_level, 1.0/(amrex::Real)_p);
        
        Print().SetPrecision(17)<<"--level "<<l<<"--"<<"\n";
        Print().SetPrecision(17)<< "L"<<_p<<" error norm:  "<<Lpnorm<<" | "<<
                            "DG Order:  "<<p+1<<" | solution component: "<<q<<"\n"; 
      }
      }
    }
  }      
}

void AmrDG::NormDG()
{
  int _p = 1;
  int N = qMp_1d;
  amrex::Vector<const amrex::MultiFab *> state_u_h(Q); 
  for(int q=0; q<Q;++q){state_u_h[q] = &(U_w[0][q]);}
  
  amrex::BoxArray c_ba = U_w[0][0].boxArray();
  
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
            u =  Initial_Condition_U(0,q,i,j,k,xi_ref_GLquad_L2proj[m]);
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
      
      Print().SetPrecision(17)<<"--level "<<0<<"--"<<"\n";
      Print().SetPrecision(17)<< "L"<<_p<<" error norm:  "<<Lpnorm<<" | "<<
                          "DG Order:  "<<p+1<<" | solution component: "<<q<<"\n"; 
    }
  }
}  


void AmrDG::L1Norm_DG_AMR()
{
  LpNorm_DG_AMR(1, xi_ref_GLquad_s,qMp_1d);
}

void AmrDG::L2Norm_DG_AMR() 
{
  //TODO:actually could generalize it to p points
  //Generate 2*(p+1) quadrature points in 1D
  int N = 2*qMp_1d;
  amrex::Vector<amrex::Real> GLquadpts;
  amrex::Real xiq = 0.0;
  amrex::Real theta = 0.0;
  for(int i=1; i<= (int)(N/2); ++i)
  {
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    if((1<=i) && (i<= (int)((1.0/3.0)*(double)N))){
      xiq = (1-0.125*(1.0/std::pow(N,2))+0.125*(1.0/std::pow(N,3))
            -(1.0/384.0)*(1.0/std::pow(N,4))*(39.0-28.0*(1.0/std::pow(std::sin(theta),2))))
            *std::cos(theta);
    }
    else if((i>(int)((1.0/3.0)*(double)N)) && (i<= (int)((double)N/2))){
      xiq = (1.0-(1.0/(8.0*std::pow((double)N,2)))
          +(1.0/(8.0*std::pow((double)N,3))))*std::cos(theta);
    }
    NewtonRhapson(xiq, N);
    GLquadpts.push_back(xiq);   
    GLquadpts.push_back(-xiq);  
  }

  //TODO: below will always be zero right?, therefore could just do GLquadpts.push_back(0.0);   
  if(N%2!=0)//if odd number, then i=1,...,N/2 will miss one value
  {
    int i = (N/2)+1;
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    xiq = (1.0-(1.0/(8.0*std::pow((double)N,2)))
          +(1.0/(8.0*std::pow((double)N,3))))*std::cos(theta);
    NewtonRhapson(xiq, N);
    GLquadpts.push_back(xiq);   
  }
  
  amrex::Vector<amrex::Vector<amrex::Real>> GLquadptsL2norm; 
  GLquadptsL2norm.resize((int)std::pow(N,AMREX_SPACEDIM),
                        amrex::Vector<amrex::Real> (AMREX_SPACEDIM));
                        
  #if (AMREX_SPACEDIM == 1)
  for(int i=0; i<N;++i)
  {
    GLquadptsL2norm[i][0]=GLquadpts[i];
  }
  #elif (AMREX_SPACEDIM == 2)
  for(int i=0; i<N;++i){
    for(int j=0; j<N;++j){
        GLquadptsL2norm[j+N*i][0]=GLquadpts[i];
        GLquadptsL2norm[j+N*i][1]=GLquadpts[j]; 
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  for(int i=0; i<N;++i){
    for(int j=0; j<N;++j){
      for(int k=0; k<N;++k){
        GLquadptsL2norm[k+N*j+N*N*i][0]=GLquadpts[i];
        GLquadptsL2norm[k+N*j+N*N*i][1]=GLquadpts[j]; 
        GLquadptsL2norm[k+N*j+N*N*i][2]=GLquadpts[k]; 
      }
    }
  }
  #endif 
  
  LpNorm_DG_AMR(2, GLquadptsL2norm,2*qMp_1d);  
}

void AmrDG::PlotFile(int tstep, amrex::Real time) const
{
  //Output AMR U_w MFab modal data for all solution components, expected 
  //to then plot the first mode,i.e cell average
  
  //using same timestep for all levels
  amrex::Vector<int> lvl_tstep; 
  for (int l = 0; l <= finest_level; ++l)
  {
    lvl_tstep.push_back(tstep);
  }
  for(int q=0; q<Q; ++q){
    amrex::Vector<std::string> plot_var_name;
    //Output all modes, then can chose which one to visualize
    //Variables naming
    
    for(int m =0 ; m<Np; ++m){
      if(sim->model_pde->equation_type == "Compressible_Euler")
      {
        if(q==0){plot_var_name.push_back("mass_density_"+std::to_string(m));}
        else if(q==1){plot_var_name.push_back("momentum_x_"+std::to_string(m));}
        else if(q==2){plot_var_name.push_back("momentum_y_"+std::to_string(m));}
        else if(q==3){plot_var_name.push_back("energy_density_"+std::to_string(m));}
        else if(q==4){plot_var_name.push_back("angular_momentum_z_"+std::to_string(m));}
      }
      else if(sim->model_pde->equation_type == "Advection")
      {
        if(q==0){
          plot_var_name.push_back("density_x_"+std::to_string(m));      
        }
      }
    }

    std::string name  = "../Results/tstep_"+std::to_string(tstep)+"_q_"+std::to_string(q)+"_plt";
    const std::string& pltfile_name = name;//amrex::Concatenate(name,5);
    
    //mf to output
    Vector<const MultiFab*> mf_out;
    for (int l = 0; l <= finest_level; ++l)
    {
      mf_out.push_back(&(U_w[l][q]));           
    }
    //amrex::WriteSingleLevelPlotfile(pltfile, U_w[q],plot_modes_name, domain_geom, time, 0);
    amrex::WriteMultiLevelPlotfile(pltfile_name, finest_level+1, mf_out, plot_var_name,
                               Geom(), time, lvl_tstep, refRatio());
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
  int q = 0;
  int lev = 1;
  int dim = 0;

  amrex::MultiFab& state_c = U_w[lev][q];
  //amrex::MultiFab& state_c = H_w[lev][q];
  //amrex::MultiFab& state_c = Fnum[lev][dim][q];
  //amrex::MultiFab& state_c = Fnumm_int[lev][dim][q];
  //amrex::MultiFab& state_c = Fp[lev][dim][q];
  //amrex::MultiFab& state_c = H_m[lev][dim][q];
  //auto ba_tmp = Fnumm_int[lev][dim][q].boxArray();
  
  Print() <<"BoxArray:  "<<state_c.boxArray()<<"\n";
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
            AllPrint() <<i<<","<<j<<"  | w="<<n<<"| "<<uc(i,j,k,n)<<"\n";
          }       
        } 
      }       
    }
  }
}

