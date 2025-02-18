#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>

using namespace amrex;

template <typename S,typename M>
class Simulation;

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

        void test(){
            //std::cout << typeid(*model_pde).name() << std::endl;
            //std::cout << typeid(model_pde).name() << std::endl;
            std::cout << "test" << std::endl;
            //model_pde->testModel();
      
          }

    protected:

        std::shared_ptr<std::ofstream> ofs;

        std::shared_ptr<NumericalMethodType> numerical_pde;

        template <typename S,typename M>
        friend class Simulation;

};


#endif 

       //void setSimulationCommunication(std::shared_ptr<Simulation<SolverType,ModelEquationType>> s){
        //    sim = s;
        //}
        
        //std::shared_ptr<Simulation<SolverType,ModelEquationType>> sim;
