#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>
//#include <variant>

template <typename EquationType>
class ModelEquation;


template <typename NumericalMethodType>
class Solver
{
    public: 
        Solver() = default;

        virtual ~Solver() = default;

        // Set reference to derived numerical method classes for communication
        void setNumericalMethod(std::shared_ptr<NumericalMethodType> nm){
            numerical_pde = nm;
        }

        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

        // Getter methods
        std::shared_ptr<NumericalMethodType> getNumericalMethod() const {
            return numerical_pde;
        }

    protected:

        std::shared_ptr<std::ofstream> ofs;

        std::shared_ptr<NumericalMethodType> numerical_pde;

        template <typename EquationType>
        friend class ModelEquation;
};


#endif 

       //void setSimulationCommunication(std::shared_ptr<Simulation<SolverType,ModelEquationType>> s){
        //    sim = s;
        //}
        
        //std::shared_ptr<Simulation<SolverType,ModelEquationType>> sim;
