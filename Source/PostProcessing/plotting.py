import yt
import pathlib
import imageio
import os
import warnings 
from yt.units import dimensions
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
warnings.filterwarnings("ignore")
yt.utilities.logger.colorize_logging()

#TODO: plot folder inside the result (Result/plots)
#TODO: currently we can do 2D slices plots of 1D,2D,3D data. Maybe  3d visualization should be implemented
#TODO: reconstruct polynomial solution in each cell and plot at itnerpolation point for most accurate visualization
#TODO: for Euler equations given energy and momentum we can derive the pressure (derived field) and this should be plotted
#TODO: isntead of custom loop over timesteps mabye use yt methods to laod entire dataset and iterate across timesteps datasets
#TODO: load AMR independently https://yt-project.org/doc/reference/api/yt.loaders.html?highlight=yt%20load#yt.loaders.load
#TODO: ideally use yt just for AMR stuff, and grid plotting. Everythign else should be done with matplotlib (easier to use imo)
#       https://yt-project.org/doc/visualizing/plots.html?highlight=matplotlib#matplotlib-customization
#       https://yt-project.org/doc/reference/api/yt.funcs.html?highlight=matplotlib#yt.funcs.matplotlib_style_context

def main_plot():
    """""---------------------------------------------""""" 
    Nsteps = 200
    
    equation_type = "Compressible_Euler_2D"
    #["Compressible_Euler_2D","Advection"]
    mode_n =0
    sol_n  = 0
    plot_every = 1
    tstep_lst = list(range(0,Nsteps+1,plot_every))
    out_name_prefix = "tstep"
    #if want to plot only one timestep or a coule of them use below
    tstep_lst = [0,1731]
    """""
    plot_every          :   plot every number of timesteps
    NB: #if output data every x timesteps better to use plot_every=1, otherwise if we output all timesteps but jsut want to plot some, use plot_every!=1
    tstep_lst            :   tstep_lst
    """""
       
    """""---------------------------------------------""""" 
      
    units_override =  {"length_unit": (1.0, "m"), "time_unit": (1.0,"s"), "mass_unit":(1.0,"kg")}
    field_composite, label_cb, label_title = get_qty_name(equation_type,sol_n,mode_n)
    max_value,min_value = get_colorbar_bounds(sol_n, Nsteps,field_composite,equation_type,sol_n,mode_n,units_override)

    print()
    print("----------------------------")

    #iterate across selected timesteps
    for tsep in tstep_lst:  
        #load data                
        file_name = "../../Results/Simulation Data/"+out_name_prefix+"_"+str(tsep)+"_q_"+str(sol_n)+"_plt"
        ds_tmp = yt.load(file_name,unit_system="mks",units_override=units_override)

        field_to_plot, field_name, ds= get_qty_to_plt(equation_type,sol_n,mode_n,ds_tmp,tsep,units_override,field_composite)     

        #######################
        #create plot
        sl = yt.SlicePlot(ds,"z", field_to_plot, origin="native")

        #Colormaps https://yt-project.org/doc/visualizing/colormaps/index.html?highlight=set_cmap
        #https://www.kennethmoreland.com/color-advice/
        #"inferno","magma","plasma","viridis","cividis"
        #"PuOr","coolwarm","RdBu","binary","gist_yarg","hot"

        
        sl.set_zlim(field=field_to_plot, zmin=min_value, zmax=max_value)
        sl.set_log(field_name, log=False)
        sl.set_cmap(field=field_name, cmap="inferno")
        sl.set_colorbar_label(field_name, label_cb)#labelpad=10
        sl.set_colorbar_minorticks("all", True)          
        sl.show_colorbar()           
          
        #AMR 
        #sl.annotate_grids(alpha=1.0, linewidth = 1.0)  
        sl.annotate_cell_edges(line_width=0.001, alpha=0.6, color='grey')

        
        #Units size,annotation and axis setings
        sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)       
        sl.annotate_title(label_title)
        #sl.set_font_size(13)
        sl.set_font_size(20)
        sl.set_axes_unit("m")    
        
        """""
        # hide the colorbar:
        sl.hide_colorbar()
        sl.hide_axes(draw_frame=True)
        
        """""
        #Output plot
        DPI = 300
        L = 5
        sl.set_buff_size(DPI*L)
        sl.set_figure_size(L) 
        
        sl.render()
        sl.save("../../Results/Plots/"+str(tsep)+"_sol_"+str(sol_n)+".png")
        
        #######################
        
def get_qty_name(equation_type,sol_n,mode_n):
    #returns the name of the variable that we want o plot in the format used to store it
    #also return the correct labels for the plot
    if equation_type == "Advection":
        if sol_n == 0:
            field_composite = "density_x_"+str(mode_n)           
            label_cb = r"$\rho(\mathbf{x})\ $ ($\dfrac{kg}{m^3}$)"  
            label_title = r"Density $\rho(\mathbf{x})$"
        
    elif equation_type == "Compressible_Euler_2D":
        if sol_n == 0:
            field_composite = "mass_density_"+str(mode_n)
            label_cb = r"$\rho(\mathbf{x})\ $ ($\dfrac{kg}{m^3}$)"   
            label_title = r"Density $\rho(\mathbf{x})$"            
        elif sol_n == 1:
            field_composite = "momentum_x_"+str(mode_n)
            label_cb = r"$u_1(\mathbf{x})\ $ ($\dfrac{m}{s}$)"   
            label_title = r"Velocity $u_1(\mathbf{x})$"
        elif sol_n == 2:
            field_composite = "momentum_y_"+str(mode_n)
            label_cb = r"$u_2(\mathbf{x})\ $ ($\dfrac{m}{s}$)"   
            label_title = r"Velocity $u_2(\mathbf{x})$"
        elif sol_n == 3:
            field_composite = "energy_density_"+str(mode_n)
            label_cb = r"$e(\mathbf{x})\ $ ($\dfrac{J}{kg}$)"  
            label_title = r"Specific Energy $e(\mathbf{x})$"
        elif sol_n == 4:
            field_composite = "angular_momentum_z_"+str(mode_n)    
            label_cb = r"$L_z(\mathbf{x})\ $ ($\dfrac{m^2}{s}$)" 
            label_title = r"Angular momentum $L_z(\mathbf{x})$"
               
    return field_composite, label_cb, label_title
                    
def get_qty_to_plt(equation_type,sol_n,mode_n,ds,t,units_override,field_composite):

    field_to_plot= ((ds.field_list)[mode_n][0],(ds.field_list)[mode_n][-1])  
    field_name = (ds.field_list)[mode_n][-1]

    #######################
    #modify data
    if equation_type == "Compressible_Euler_2D":
        if sol_n!=0:
            #add density field
            density_file_name = "../../Results/Simulation Data/tstep_"+str(t)+"_q_"+str(0)+"_plt"
            density_ds = yt.load(density_file_name,unit_system="mks",units_override=units_override)
            
            def _field_mode_density(field,data):
                return  density_ds.r[density_field_name]*density_ds.unit_system["density"]
            
            def _field_mode_velocity_d(field,data):
                momentum_field_name = ((ds.field_list)[mode_n][0],(ds.field_list)[mode_n][-1])                  
                return ((ds.r[momentum_field_name])*(ds.unit_system["momentum"]/ds.unit_system["volume"])/ds.r[density_field_name])
                
            def _field_mode_energy(field,data):
                energy_density_field_name = ((ds.field_list)[mode_n][0],(ds.field_list)[mode_n][-1])    
                return ((ds.r[energy_density_field_name])*(ds.unit_system["specific_energy"]*ds.unit_system["density"])/ds.r[density_field_name])
                
            density_field_name = ((density_ds.field_list)[mode_n][0],(density_ds.field_list)[mode_n][-1])
            ds.add_field(name=density_field_name, function=_field_mode_density, sampling_type="cell", dimensions=dimensions.density,units=ds.unit_system["density"])
            
            #divide momentum by density and store it as derived field
            if sol_n==1 or sol_n==2:
                velocity_field_composite=field_composite.replace("momentum", "velocity")                    
                velocity_derived_field_name = ((ds.field_list)[mode_n][0],velocity_field_composite)
                ds.add_field(name=velocity_derived_field_name, function=_field_mode_velocity_d, sampling_type="cell",dimensions=dimensions.velocity,units=ds.unit_system["velocity"])

                field_to_plot=velocity_derived_field_name
                field_name=velocity_field_composite
            elif sol_n==3:
                energy_field_composite=field_composite.replace("density_", "")                    
                energy_derived_field_name = ((ds.field_list)[mode_n][0],energy_field_composite)
                ds.add_field(name=energy_derived_field_name, function=_field_mode_energy,
                            sampling_type="cell",dimensions=dimensions.specific_energy,units=ds.unit_system["specific_energy"])

                field_to_plot=energy_derived_field_name
                field_name=energy_field_composite
                
    return field_to_plot, field_name, ds

def get_colorbar_bounds(q, Nsteps,field_composite,equation_type,sol_n,mode_n,units_override):
    #construct colorbar (need to find max/min values in order to normalize it
    #to speed up we look at values one_every files
    min_val_steps = []
    max_val_steps = []
    one_every = 50

    for tsep in range(0,Nsteps+1,one_every):
        try:
            #load data
            file_name = "../../Results/Simulation Data/tstep_"+str(tsep)+"_q_"+str(q)+"_plt"
            ds_tmp = yt.load(file_name,unit_system="mks")
            field_to_plot, field_name, ds= get_qty_to_plt(equation_type,sol_n,mode_n,ds_tmp,tsep,units_override,field_composite)
            #get max/min
            mf =ds.r[field_name] 
            #tmp_min_val = mf.min()
            #tmp_max_val = mf.max()
            tmp_min_val = ds.all_data().quantities.extrema(field_name)[0]
            tmp_max_val = ds.all_data().quantities.extrema(field_name)[1]
            min_val_steps.append(tmp_min_val)
            max_val_steps.append(tmp_max_val)    
        except Exception as e:
            print(e)
            continue
    
    max_value = max(max_val_steps)
    min_value = min(min_val_steps)

    return max_value, min_value
                       
if __name__ == "__main__":
    main_plot()
