#include <AMReX.H>
#include <iostream>
#include "Simulation.h"
#include "ModelEquation.h"

using namespace amrex;

int main(int argc, char* argv[])
{    
    amrex::Initialize(argc,argv);
    {  
      Simulation sim;
            
      //Solved equation
      std::string equation_type = "Compressible_Euler";
      std::string test_case = "isentropic_vortex";
      bool source_term = false;
      bool angular_momentum = false;      
      
      sim.settings_case(equation_type, source_term,angular_momentum, test_case);
      /*-------------------------------------------*/
      //Numeric Settings
      //AMR/MISC
      int max_level = 0;// number of levels = max_level + 1
      int t_regrid  = 1;// try regrid every t_regrid timesteps
      int t_limit   = 1;//every how many timesteps apply limtier
            
      //Limiters
      //  TVB based
      std::string limiter_type = "TVB";
      amrex::Real TVB_M = 20.0; //constant parameter used on minmod bounded definition
      //the smaller TVB_M is, the easier it is for a cell to be tagged for refinement
      //can happen that this value is too
      
      //Time integration scheme
      amrex::Real T =  2.0; 
      amrex::Real CFL = 1.0;//safety factor for time step, can decrease the timestep
      
      //DG settings (order p+1)
      int p  = 3; //order of basis function
     
      //I/O settings
      int t_outplt = 10;
      
      /*-------------------------------------------*/
      //DOMAIN SETTINGS
      int n_cell_x = 16;
      int n_cell_y = 16;
      int n_cell_z = 16;

      amrex::Real L_x_lo   = 0.0;
      amrex::Real L_x_hi   = 2.0;
      
      amrex::Real L_y_lo   = 0.0;
      amrex::Real L_y_hi   = 2.0;
      
      amrex::Real L_z_lo   = 0.0;
      amrex::Real L_z_hi   = 2.0;
      
      if(test_case == "kelvin_helmolz_instability")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 1.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 1.0;
        
        n_cell_x = 32;
        n_cell_y = 32;
        T = 2.0;
        CFL = 1.0;
        TVB_M = 1.0;
        max_level = 2;
        p  = 4;
        //p=4,max_level=4
        //lmax tvb_l=50
        //TVB_M=[15,40,60]
        //p=4
      }
      else if(test_case == "isentropic_vortex" || test_case == "isentropic_vortex_static")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 10.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 10.0; 
        
        n_cell_x = 16;
        n_cell_y = 16;
        
        TVB_M = 0.0;
        p  = 3;
        CFL = 0.9;
        T = 10.0;
        max_level = 1;
        t_limit =0;
        t_outplt = 0;
        t_regrid  = 1;
      }
      else if(test_case ==  "richtmeyer_meshkov_instability")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 40.0/3.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 40.0;  
        
        n_cell_x = 32;
        n_cell_y = 96;
           
        CFL = 0.9;
        p  = 2;
        T = 50.0;
        
        t_outplt = 1;
        t_limit = 1;
        t_regrid  = 1;
        max_level = 0;
        TVB_M = 0.1;
        
        //working settings
        // TVB_M = 0.1;p  = 2;max_level = 1;
      }
      else if(test_case == "keplerian_disc")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 6.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 6.0;   
        
        n_cell_x = 64;
        n_cell_y = 64;
        
        p  = 2;
        CFL = 0.5;
        T = 120.0;
        max_level = 0;
        t_outplt = 1;
        TVB_M = 1.0;
        source_term = true;
        angular_momentum = true; 
        t_limit =1;
        t_outplt = 1;
        t_regrid  = 1;
      }
      else if(test_case=="gaussian_shape")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 2.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 2.0;   
        
        n_cell_x = 16;
        n_cell_y = 16;  
        
        p  = 3;
        CFL = 1.0;
        T = 2.0;
        max_level = 1;
        TVB_M = 1.0;
        source_term = false;
        angular_momentum = false; 
        t_limit =0;
        t_outplt = 10;
        t_regrid  = 1;
      }

      //AMR tagging criteria
      //NB: certain refinement criteria induce an extra mesh iteration, 
      //therefore are more heavy computationally
            
      //Default AMR refinement criteria based on value
      amrex::Vector<amrex::Real> AMR_C(max_level+1);
      for(int l=0; l<max_level+1;++l)
      {
        //code here if you want different coefficients for each level of refinement
        //AMR_C[l] = 1.0+l*0.25; 
        if(l==0){AMR_C[l] = 0.7;}
        else if(l==1){AMR_C[l] = 1.7;}
        else if(l==2){AMR_C[l] = 1.8;}
      }
 
      //Velocity curl (3D only)
      amrex::Vector<amrex::Real> AMR_curl_C(max_level+1);
      for(int l=0; l<max_level+1;++l)
      {      
        AMR_curl_C[l] = 1.0+l*0.25;        
      }
    
      //Velocity Divergence
      amrex::Vector<amrex::Real> AMR_div_C(max_level+1);
      for(int l=0; l<max_level+1;++l)
      {
        AMR_div_C[l] = 1.0+l*0.25;        
      }
      
      //Density Gradient
      amrex::Vector<amrex::Real> AMR_grad_C(max_level+1);
      for(int l=0; l<max_level+1;++l)
      {
        AMR_grad_C[l] = 1.0+l*0.25;        
      }
           
      //Second derivative (make sense only for p>=2)
      amrex::Real AMR_sec_der_indicator = 0.5;
      amrex::Vector<amrex::Real> AMR_sec_der_C(max_level+1);
      for(int l=0; l<max_level+1;++l)
      {
        //AMR_sec_der_C[l] = 1.0+l*0.25;        
        AMR_sec_der_C[l] =1.0;
      }
      
      //  TVB based refinement->modify under limiters
      amrex::Vector<amrex::Real> AMR_TVB_C(max_level+1);
      for(int l=0; l<max_level+1;++l)
      {
        if(l==0){AMR_TVB_C[l]=1e-6;}
        else if(l==1){AMR_TVB_C[l]=1e-3;}
        else if(l==2){AMR_TVB_C[l]=0.1;}
        //AMR_TVB_C[l] = 1.0+(amrex::Real)l*2.0;        
      }

      amrex::Vector<int> n_cell{AMREX_D_DECL(n_cell_x, n_cell_y, n_cell_z)};
      amrex::RealBox real_box {{AMREX_D_DECL(L_x_lo,L_y_lo,L_z_lo)}, 
                              {AMREX_D_DECL(L_x_hi, L_y_hi, L_z_hi)}}; 
      amrex::Vector<amrex::IntVect> refinement;
      for (int l = 0; l < max_level; ++l) {
        refinement.push_back(amrex::IntVect(AMREX_D_DECL(2, 2, 2)));
        //don't change ref criteria, some functions expect refinement 2
      }
      
      /*-------------------------------------------*/
      //Boundary Conditions
      int Q = (sim.model_pde)->Q_model;      
            
      amrex::Array<int,AMREX_SPACEDIM> is_periodic;
      
      amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> bc_lo;
      amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> bc_hi;
      bc_lo.resize(Q);
      bc_hi.resize(Q);
      amrex::Vector<amrex::Vector<int>> bc_lo_type;
      amrex::Vector<amrex::Vector<int>> bc_hi_type;
      bc_lo_type.resize(Q);
      bc_hi_type.resize(Q);
      
      //either all solution components are periodic or they are not
      #if(AMREX_SPACEDIM == 1)
        is_periodic[0]  = 1;
        
        for(int q=0; q<Q; ++q)
        {
          bc_lo_type[q].resize(AMREX_SPACEDIM);
          bc_hi_type[q].resize(AMREX_SPACEDIM);
          
          bc_lo[q][0]=BCType::int_dir;
         
          bc_hi[q][0]=BCType::int_dir;
          
          bc_lo_type[q][0]=0;
        
          bc_hi_type[q][0]=0;        
        }

      #elif(AMREX_SPACEDIM == 2)
        is_periodic[0]  = 1;
        is_periodic[1]  = 1;
        
        for(int q=0; q<Q; ++q)
        {       
          bc_lo_type[q].resize(AMREX_SPACEDIM);
          bc_hi_type[q].resize(AMREX_SPACEDIM);
          
          bc_lo[q][0]=BCType::int_dir;
          bc_lo[q][1]=BCType::int_dir;
          
          bc_hi[q][0]=BCType::int_dir;
          bc_hi[q][1]=BCType::int_dir;
          
          bc_lo_type[q][0]=1;
          bc_lo_type[q][1]=1;
          
          bc_hi_type[q][0]=1;
          bc_hi_type[q][1]=1;
        }
        if(test_case ==  "richtmeyer_meshkov_instability")
        {
          //need periodic in x to ensure waves interaction
          is_periodic[0]  = 0;
          is_periodic[1]  = 0;
          for(int q=0; q<Q; ++q)//rho,rhou1,rhou2,rhoe
          {   
            //Neumann BC everywhere
            bc_lo[q][1]=BCType::ext_dir;
            bc_hi[q][1]=BCType::ext_dir;
            bc_lo[q][0]=BCType::ext_dir;
            bc_hi[q][0]=BCType::ext_dir;
            
            bc_lo_type[q][0]=1;
            bc_lo_type[q][1]=1;
            
            bc_hi_type[q][0]=1;
            bc_hi_type[q][1]=1;
          }   
        }
        else if(test_case == "double_mach_reflection")
        {
          is_periodic[0]  = 1;
          is_periodic[1]  = 1;
          for(int q=0; q<Q; ++q)//rho,rhou1,rhou2,rhoe
          {   
            //Neumann BC everywhere
            bc_lo[q][1]=BCType::int_dir;
            bc_hi[q][1]=BCType::int_dir;
            bc_lo[q][0]=BCType::int_dir;
            bc_hi[q][0]=BCType::int_dir;
            
            bc_lo_type[q][0]=1;
            bc_lo_type[q][1]=1;
            
            bc_hi_type[q][0]=1;
            bc_hi_type[q][1]=1;
          }           
        }
        else if(test_case ==  "isentropic_vortex")
        {
          //need periodic in x to ensure waves interaction
          is_periodic[0]  = 1;
          is_periodic[1]  = 1;
          for(int q=0; q<Q; ++q)//rho,rhou1,rhou2,rhoe
          {   
            //Neumann BC everywhere
            bc_lo[q][1]=BCType::int_dir;
            bc_hi[q][1]=BCType::int_dir;
            bc_lo[q][0]=BCType::int_dir;
            bc_hi[q][0]=BCType::int_dir;
            
            bc_lo_type[q][0]=1;
            bc_lo_type[q][1]=1;
            
            bc_hi_type[q][0]=1;
            bc_hi_type[q][1]=1;
          }   
        }
      #elif(AMREX_SPACEDIM == 3)
        is_periodic[1]  = 1;
        is_periodic[2]  = 1;
        for(int q=0; q<Q; ++q)
        { 
          bc_lo_type[q].resize(AMREX_SPACEDIM);
          bc_hi_type[q].resize(AMREX_SPACEDIM);
          
          bc_lo[q][0]=BCType::ext_dir;
          bc_lo[q][1]=BCType::ext_dir;
          bc_lo[q][2]=BCType::ext_dir;
          
          bc_hi[q][0]=BCType::ext_dir;
          bc_hi[q][1]=BCType::ext_dir;
          bc_hi[q][2]=BCType::ext_dir;
          
          bc_lo_type[q][0]=1;
          bc_lo_type[q][1]=1;
          bc_lo_type[q][2]=1;
          
          bc_hi_type[q][0]=1;
          bc_hi_type[q][1]=1;
          bc_hi_type[q][2]=1;
        }
      #endif
      //NB: BCType::int_dir==>Periodic
      //    BCType::ext_dir==>Dirichlet/Neumann
      
      //  AMREX provides its own boundary conditions types, also for Neumann (), 
      //but I did my own 
      //  implementation so that we can apply it to the modes and its easier 
      //to apply to our equations
      //  Differenciate w.r.t bc types using bc_lo/hi_type
      //    "dirichlet" == 0
      //    "neumann"   == 1
      
      sim.settings_general(real_box,max_level,n_cell, refinement,
                    is_periodic,bc_lo,bc_hi,bc_lo_type,bc_hi_type,T,CFL,p,
                    t_regrid,t_outplt,limiter_type,TVB_M,
                    AMR_TVB_C, AMR_curl_C, AMR_div_C, AMR_grad_C, 
                    AMR_sec_der_C,AMR_sec_der_indicator, AMR_C, t_limit);
                    
      /*-------------------------------------------*/
      // wallclock time
      const auto strt_total = amrex::second();
                          
      sim.run();
                    
      // wallclock time
      auto end_total = amrex::second() - strt_total;
      
      // print wallclock time
      ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
      amrex::Print() << "\nTotal Time: " << end_total << '\n';
    }
    amrex::Finalize();
    
    return 0;
}

/*------------------------------------------------------------------------*/
/* 
--GENERAL CONSIDERATIONS--
TODO: at level L skip covered cells, just then send data fluxes from fine to 
      coarse, if

TODO: function for most commonly used AMR settings, and abiltiy to have presets
TODO: not use global timestepping, but individaul timestepping for each level
TODO: variable naming convention, constructor convention ,...
TODO: limiter class, refinement class
TODO: future, take inspiration from AmrLevel class to extend/modify current code

TODO:-computing ghost values vs exchanging them?
      curently each rhank computes its own ghsot cells values, what about not 
      computing thema dn just exchange them at the end?THis can be done for
      fluxes before Interfacialflux. The important thing are not the internal ones 
      but the boundary ones, we need them them to be updated with the predictor
      We

HPC Considerations:
TODO: AMD PRAGMA SIMD-> compiler directive for vectorization, should be added to 
      innermost loops
TODO: Ordering of varialbes in vector and Mfabs$
TODO: efficient evaluation of Phi(xi) and DPhi(xi) at all nodes uisng iterative method
*/
/*------------------------------------------------------------------------*/
/*
      else if(test_case == "double_mach_reflection")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 3.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 1.0;   
        
        n_cell_x = 160;
        n_cell_y = 48;  
        
        CFL = 0.1;
        p  = 1;
        T = 1.0;
        
        t_outplt = 10;
        t_limit = 1;
        t_regrid  = 1;
        
        max_level = 2;
        TVB_M = 1.0;      
      }*/
/*

      else if(test_case == "radial_shock_tube")
      {
        L_x_lo   = -1.0;
        L_x_hi   = 1.0;
      
        L_y_lo   = -1.0;
        L_y_hi   = 1.0; 
        
        n_cell_x = 32;
        n_cell_y = 32;
        T = 0.5;
        CFL = 1.0;
        TVB_M = 1;
        max_level = 2;
        p  = 3;
      }
      else if(test_case == "keplerian_disc")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 6.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 6.0;   
        
        n_cell_x = 128;
        n_cell_y = 128;
        
        p  = 1;
        CFL = 0.5;
        T = 120.0;
        max_level = 0;
      }
      */
