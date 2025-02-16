#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>
#include <variant>

//class Simulation;
//#include "Simulation.h"
//NB: ideal would be forward decl here and in numericalmethod implementation
//where we use sim ptr actually import the Sim header. But in this way user might forget,
//therefore already doit here.

//tempalte is usefull because in this way we can automatically
//create a pointer to the correct derived class inside Solver
template <typename NumericalMethodType>//, typename U = std::shared_ptr<void>
class Solver
{
    public: 
        Solver() : model_pde(nullptr) {};

        virtual ~Solver() = default;

        // Set reference to derived numerical method classes for communication
        void setNumericalMethod(std::shared_ptr<NumericalMethodType> nm){
            numerical_pde = nm;
        }

        template <typename M>
        void setModelEquation(M&& m)
        {
            //model_pde = std::shared_ptr<U>(std::forward<M>(m));
            //model_pde = std::static_pointer_cast<void>(std::make_shared<std::decay_t<M>>(std::forward<M>(m)));
            model_pde = std::make_shared<std::decay_t<M>>(std::forward<M>(m));
        }

        

        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

    protected:

        std::shared_ptr<std::ofstream> ofs;

        std::shared_ptr<NumericalMethodType> numerical_pde;

        std::shared_ptr<void> model_pde;

};


#endif 

       //void setSimulationCommunication(std::shared_ptr<Simulation<SolverType,ModelEquationType>> s){
        //    sim = s;
        //}
        
        //std::shared_ptr<Simulation<SolverType,ModelEquationType>> sim;
