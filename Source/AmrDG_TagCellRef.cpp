#include "AmrDG.h"

//NB: uw needs to be a const MFab in this function
void AmrDG::AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
  //Print(sim->ofs) <<"AmrDG::ErrorEst"<<"\n";
  //check and flag cells where regridding criteria is met
  const int   tagval = TagBox::SET;
   
  /*
  amrex::Vector<amrex::MultiFab> tmp_U_p(Q);
  amrex::Vector<amrex::MultiFab> tmp_U_m(Q);
  for(int q=0 ; q<Q; ++q){
    tmp_U_p[q].define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), qMpbd, nghost);
    tmp_U_p[q].setVal(0.0);
    
    tmp_U_m[q].define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), qMpbd, nghost);
    tmp_U_m[q].setVal(0.0);
  }
 
  amrex::MultiFab& state_curl_indicator =idc_curl_K[lev];
  amrex::MultiFab& state_div_indicator =idc_div_K[lev];
  amrex::MultiFab& state_grad_indicator =idc_grad_K[lev];
  

  bool any_trouble = false;//flag used to indicate if any troubled cells have been found at all

  amrex::Vector<amrex::MultiFab *> state_tmp_um(Q);
  amrex::Vector<amrex::MultiFab *> state_tmp_up(Q);
  */
  amrex::Vector<const amrex::MultiFab *> state_uw(Q);

  for(int q=0; q<Q; ++q){
    //state_tmp_um[q]=&(tmp_U_m[q]);
    //state_tmp_up[q]=&(tmp_U_p[q]);
    state_uw[q]=&(U_w[lev][q]); 
  }
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    /*
    amrex::Vector< amrex::FArrayBox *> fab_tmp_um(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > tmp_um(Q);
    amrex::Vector< amrex::FArrayBox *> fab_tmp_up(Q);
    amrex::Vector< amrex::Array4<  amrex::Real> > tmp_up(Q);
    */
    amrex::Vector<const amrex::FArrayBox *> fab_uw(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > uw(Q);   

    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_uw[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_uw[0]),true); mfi.isValid(); ++mfi)
    #endif   
    {
      const amrex::Box& bx = mfi.tilebox();  
      const auto tagfab  = tags.array(mfi);

      /*
      amrex::FArrayBox& fab_curl_indicator = state_curl_indicator[mfi];
      amrex::FArrayBox& fab_div_indicator = state_div_indicator[mfi];
      amrex::FArrayBox& fab_grad_indicator = state_div_indicator[mfi];*/
          
      for(int q=0 ; q<Q; ++q){
        //fab_tmp_um[q] = &((*(state_tmp_um[q]))[mfi]);
        //fab_tmp_up[q] = &((*(state_tmp_up[q]))[mfi]);
        fab_uw[q] = state_uw[q]->fabPtr(mfi);
        
        //tmp_um[q] = (*(fab_tmp_um[q])).array();
        //tmp_up[q] = (*(fab_tmp_up[q])).array();
        uw[q] = fab_uw[q]->const_array();
      }
              
      //amrex::Array4<Real> const& curl_indicator = fab_curl_indicator.array();
      //amrex::Array4<Real> const& div_indicator = fab_div_indicator.array();
      //amrex::Array4<Real> const& grad_indicator = fab_grad_indicator.array();
      
      amrex::Dim3 lo = lbound(bx);
      amrex::Dim3 hi = ubound(bx);
      //AMR_curl_indicator = 0.0;
      //AMR_div_indicator = 0.0;
      
      amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
      {
        //AMRIndicator_tvb(i, j, k, &uw, &tmp_um,&tmp_up, lev, tagfab, tagval, any_trouble);
        //AMRIndicator_curl(i, j, k, &uw, curl_indicator, lev, true , tagfab, tagval, any_trouble);
        //AMRIndicator_div(i, j, k, &uw, div_indicator, lev, true , tagfab, tagval, any_trouble);
        //AMRIndicator_grad(i, j, k, &uw, grad_indicator, lev, true , tagfab, tagval, any_trouble);
        //AMRIndicator_second_derivative(i, j, k,&uw, lev , tagfab, tagval, any_trouble);
        //if(uw[0](i,j,k,0)<=AMR_C[lev])
        if(uw[0](i,j,k,0)<=1.0)//AMR_C[lev]
        {
          tagfab(i,j,k)=tagval;
        }
      });
      /*
      //TODO: execute below only if using curl_euler,div_euler or grad_euler
        #ifdef AMREX_USE_OMP
        #pragma omp single if(omp_get_thread_num() ==0)
        #endif
        {
          //Get the number of cells in each dimension of the domain
          int Nc= (int)CountCells(lev);//might have problems if we have more than 2 billion cells
                   
          if(equation_type == "2d_euler" || equation_type == "2d_euler_am"){
            AMR_div_indicator=std::sqrt(AMR_div_indicator/Nc);
            AMR_grad_indicator=std::sqrt(AMR_div_indicator/Nc);
          }
          else if(equation_type == "3d_euler" || equation_type == "3d_euler_am"){
            AMR_curl_indicator=std::sqrt(AMR_curl_indicator/Nc);
            AMR_div_indicator=std::sqrt(AMR_div_indicator/Nc);
            AMR_grad_indicator=std::sqrt(AMR_div_indicator/Nc);
          }
        }
        
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {          
          AMRIndicator_curl(i, j, k, &uw, curl_indicator, lev, true , tagfab, tagval, any_trouble);
          AMRIndicator_div(i, j, k, &uw, div_indicator, lev, true , tagfab, tagval, any_trouble);
          AMRIndicator_grad(i, j, k, &uw, grad_indicator, lev, true , tagfab, tagval, any_trouble);
        });
        */
    }
  }
}
    
/*
void AmrDG::AMRIndicator_tvb(int i, int j, int k,
                            amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                            amrex::Vector<amrex::Array4<amrex::Real>>* um,
                            amrex::Vector<amrex::Array4<amrex::Real>>* up,int l, 
                            amrex::Array4<char> const& tag,char tagval,
                            bool& any_trouble)
{
  amrex::Vector<amrex::Vector<amrex::Real>> L_EV;
  
  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    if(tag(i,j,k) != tagval)//done to avoid repeating computation in higher 
    //dimensions. I.e if we tag cell when iterating in x, then no need to further doit in y,z
    {
      for(int m = 0; m<qMpbd ; ++m){
        get_u_from_u_w(m, i, j, k,uw, um, xi_ref_GLquad_bdm[d][m]);
        get_u_from_u_w(m, i, j, k,uw, up, xi_ref_GLquad_bdp[d][m]);
      }
      
      amrex::Real Dm_u_avg;
      amrex::Real Dp_u_avg;
      amrex::Real Dm_u;
      amrex::Real Dp_u;
      amrex::Real v;
      int shift[] = {0,0,0};
      shift[d] = 1;
      
      for(int q=0; q<Q_unique; ++q)
      {
        for(int m = 0; m<qMpbd ; ++m){ 
          L_EV = model_pde->pde_EV_Lmatrix(d,0,i,j,k,uw);

          Dm_u_avg = 0.0;
          Dp_u_avg = 0.0;
          Dm_u = 0.0;
          Dp_u = 0.0;
          v = 0.0;
          bool troubled_flag_m =false;
          bool troubled_flag_p =false;
          for(int _q=0; _q<Q_unique; ++_q)//in scalar case sim->model_pde->L_EV[q][_q]==1.0;
          {       
            Dm_u_avg += L_EV[q][_q]*(((*uw)[_q])(i,j,k,0)-
                                      ((*uw)[_q])(i-shift[0],j-shift[1],k-shift[2],0));
                                      
            Dp_u_avg += L_EV[q][_q]*(((*uw)[_q])(i+shift[0],j+shift[1],k+shift[2],0)
                                    -((*uw)[_q])(i,j,k,0));        
                                    
            Dm_u += L_EV[q][_q]*(((*uw)[_q])(i,j,k,0)-((*um)[_q])(i,j,k,m));
            
            Dp_u += L_EV[q][_q]*(((*up)[_q])(i,j,k,m)-((*uw)[_q])(i,j,k,0));            
          }

          amrex::Real corrm = minmodB(Dm_u,Dm_u_avg,Dp_u_avg, troubled_flag_m, l);
          
          amrex::Real corrp = minmodB(Dp_u,Dm_u_avg,Dp_u_avg, troubled_flag_p, l);
          
          if(troubled_flag_m && troubled_flag_p){
            tag(i,j,k) = tagval;
            any_trouble = true;
            break;
          }
        }
      }
      shift[d] = 0; //not necessary    
    }   
  }
}

void AmrDG::get_u_from_u_w(int c, int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                          amrex::Vector<amrex::Array4< amrex::Real>>* u ,
                          amrex::Vector<amrex::Real> xi)
{
  //computes the sum of modes and respective basis function evaluated at specified location
  //for all solution components
  for(int q=0 ; q<Q; ++q){
    amrex::Real sum = 0.0;
    for (int n = 0; n < Np; ++n){  
      sum+=(((*uw)[q])(i,j,k,n)*Phi(n, xi));
    }
    ((*u)[q])(i,j,k,c) = sum;
  }
}

void AmrDG::AMRIndicator_second_derivative(int i, int j, int k,
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                                          int l,amrex::Array4<char> const& tag,char tagval, 
                                          bool& any_trouble)
{

  int q;
  //int q = sim->model_pde->amr_second_derivative_component;//TODO
  
  auto const dx = geom[l].CellSizeArray();  
  
  amrex::Vector<amrex::Real> xi_ref_center(AMREX_SPACEDIM);
  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    xi_ref_center[d]=0.0;
  }
  
  amrex::Real nominator=0.0;
  amrex::Real denominator = 0.0;
  amrex::Real epsilon = 1e-3;
  for(int d1=0; d1<AMREX_SPACEDIM; ++d1)
  {
    int shift[] = {0,0,0};
    for(int d2=0; d2<AMREX_SPACEDIM; ++d2)
    {
      shift[d2] = 1;
      amrex::Real tmp_sec_der = model_pde->get_D2U_from_U_w(d1,d2,q,i, j, k,uw,xi_ref_center);
      
      nominator+=std::pow(tmp_sec_der,2.0);

      denominator+=std::pow((((std::abs(model_pde->get_DU_from_U_w(d1, q, 
                            i+shift[0], j+shift[1],  k+shift[2],uw,xi_ref_center))+std::abs(
                            model_pde->get_DU_from_U_w(d1, q,  i, 
                            j,  k,uw,xi_ref_center))) /(dx[d2]))+epsilon*std::abs(tmp_sec_der)),2.0);
      shift[d2] = 0; 
    }  
  }
  amrex::Real second_derivative_indicator = std::sqrt(nominator/denominator);
  
  if(second_derivative_indicator>AMR_sec_der_indicator*AMR_sec_der_C[l])
  {
    tag(i,j,k) = tagval;
    any_trouble = true;
  }   
}

void AmrDG::AMRIndicator_curl(int i, int j, int k,
                              amrex::Vector<amrex::Array4< const amrex::Real>>* uw, 
                              amrex::Array4<amrex::Real> const & curl_indicator,int l, 
                              bool flag_local,amrex::Array4<char> const& tag,
                              char tagval,bool& any_trouble)
{
  //VELOCITY CURL BASED CELL INDICATOR
  if(flag_local)
  {
    auto const dx = geom[l].CellSizeArray();   
    amrex::Real vol = 1.0;
    for(int d=0; d<AMREX_SPACEDIM; ++d){vol*=dx[d];}   
    amrex::Real di = std::pow(vol, 1.0/AMREX_SPACEDIM);
    
    amrex::Vector<amrex::Real> xi_ref_center(AMREX_SPACEDIM);
    for  (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_ref_center[d]=0.0;
    }
    
    amrex::Real ddxu2 = model_pde->get_DU_from_U_w(0,2,i,j,k,uw,xi_ref_center);
    amrex::Real ddxu3 = model_pde->get_DU_from_U_w(0,3,i,j,k,uw,xi_ref_center);
    amrex::Real ddyu1 = model_pde->get_DU_from_U_w(1,1,i,j,k,uw,xi_ref_center);
    amrex::Real ddyu3 = model_pde->get_DU_from_U_w(1,3,i,j,k,uw,xi_ref_center);
    amrex::Real ddzu1 = model_pde->get_DU_from_U_w(2,1,i,j,k,uw,xi_ref_center);
    amrex::Real ddzu2 = model_pde->get_DU_from_U_w(2,2,i,j,k,uw,xi_ref_center);
    
    amrex::Real curl = std::sqrt(std::pow(ddyu3-ddzu2,2.0)+std::pow(ddzu1-ddxu3,2.0)+std::pow(ddxu2-ddyu1,2.0)); 
    curl_indicator(i,j,k,0)= curl*std::pow(di,3.0/2.0);
    AMR_curl_indicator+=(std::pow(curl_indicator(i,j,k,0),2.0));
  }
  else
  {
    //this part is executed after we hae already iterated across all mesh once 
    //and constructed all MFabs containing the curl indicator for each cell
    //here we compare each cell indicator w.r.t the global one (global one 
    //obtained by weighted sum of lcoal ones , done during first mesh iteration
    //global one stored as class variable
    if(curl_indicator(i,j,k,0)>AMR_curl_indicator*AMR_curl_C[l])
    {
      tag(i,j,k) = tagval;
      any_trouble = true;
    }
  }
}

void AmrDG::AMRIndicator_div(int i, int j, int k,
                            amrex::Vector<amrex::Array4< const amrex::Real>>* uw, 
                            amrex::Array4<amrex::Real> const & div_indicator,int l, 
                            bool flag_local,amrex::Array4<char> const& tag,
                            char tagval,bool& any_trouble)
{
  //VELOCITY DIVERGENCE BASED CELL INDICATOR
  if(flag_local)
  {
    auto const dx = geom[l].CellSizeArray();   
    amrex::Real vol = 1.0;
    for(int d=0; d<AMREX_SPACEDIM; ++d){vol*=dx[d];}   
    amrex::Real di = std::pow(vol, 1.0/AMREX_SPACEDIM);
    
    amrex::Vector<amrex::Real> xi_ref_center(AMREX_SPACEDIM);
    for  (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_ref_center[d]=0.0;
    }
    #if (AMREX_SPACEDIM == 2)
      amrex::Real ddxu1 = model_pde->get_DU_from_U_w(0,1,i,j,k,uw,xi_ref_center);
      amrex::Real ddyu2 = model_pde->get_DU_from_U_w(1,2,i,j,k,uw,xi_ref_center);
      amrex::Real div = std::abs(ddxu1+ddyu2);    
      
      div_indicator(i,j,k,0)= div*std::pow(di,3.0/2.0);
      AMR_div_indicator+=(std::pow(div_indicator(i,j,k,0),2.0));
    #elif (AMREX_SPACEDIM == 3)
      amrex::Real ddxu1 = model_pde->get_DU_from_U_w(0,1,i,j,k,uw,xi_ref_center);
      amrex::Real ddyu2 = model_pde->get_DU_from_U_w(1,2,i,j,k,uw,xi_ref_center);
      amrex::Real ddzu3 = model_pde->get_DU_from_U_w(2,3,i,j,k,uw,xi_ref_center);      
      amrex::Real div = std::abs(ddxu1+ddyu2+ddzu3);
      
      div_indicator(i,j,k,0)= div*std::pow(di,3.0/2.0);
      AMR_div_indicator+=(std::pow(div_indicator(i,j,k,0),2.0));
    #endif
  }
  else
  {
    if(div_indicator(i,j,k,0)>AMR_div_indicator*AMR_div_C[l])
    {
      tag(i,j,k) = tagval;
      any_trouble = true;
    }
  }
}

void AmrDG::AMRIndicator_grad(int i, int j, int k,
                              amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                              amrex::Array4<amrex::Real> const & grad_indicator,
                              int l, bool flag_local,amrex::Array4<char> const& tag,
                              char tagval,bool& any_trouble)
{
  //DENSITY GRADIENT BASED CELL INDICATOR
  if(flag_local)
  {
    auto const dx = geom[l].CellSizeArray();   
    amrex::Real vol = 1.0;
    for(int d=0; d<AMREX_SPACEDIM; ++d){vol*=dx[d];}   
    amrex::Real di = std::pow(vol, 1.0/AMREX_SPACEDIM);
    
    amrex::Vector<amrex::Real> xi_ref_center(AMREX_SPACEDIM);
    for  (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_ref_center[d]=0.0;
    }
  
    #if (AMREX_SPACEDIM == 2)    
    amrex::Real ddx_rho = model_pde->get_DU_from_U_w(0,0,i,j,k,uw,xi_ref_center);
    amrex::Real ddy_rho = model_pde->get_DU_from_U_w(1,0,i,j,k,uw,xi_ref_center);
    
    amrex::Real grad = std::sqrt(std::pow(ddx_rho,2)+std::pow(ddy_rho,2));
    
    grad_indicator(i,j,k,0)= grad*std::pow(di,3.0/2.0);
    AMR_grad_indicator+=(std::pow(grad_indicator(i,j,k,0),2));
    #elif (AMREX_SPACEDIM == 3)
    amrex::Real ddx_rho = model_pde->get_DU_from_U_w(0,0,i,j,k,uw,xi_ref_center);
    amrex::Real ddy_rho = model_pde->get_DU_from_U_w(1,0,i,j,k,uw,xi_ref_center);
    amrex::Real ddz_rho = model_pde->get_DU_from_U_w(2,0,i,j,k,uw,xi_ref_center);
    
    amrex::Real grad = std::sqrt(std::pow(ddx_rho,2.0)+std::pow(ddy_rho,2.0)
                                +std::pow(ddz_rho,2.0));    
    
    grad_indicator(i,j,k,0)= grad*std::pow(di,3.0/2.0);
    AMR_grad_indicator+=(std::pow(grad_indicator(i,j,k,0),2.0));
    #endif
  }
  else
  {
    if(grad_indicator(i,j,k,0)>AMR_grad_indicator*AMR_grad_C[l])
    {
      tag(i,j,k) = tagval;
      any_trouble = true;
    }  
  } 
}
*/